import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import math
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from backend.utils.config import MODEL_PARAMS

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ProbAttention(nn.Module):
    """
    ProbSparse Attention mechanism from Informer paper.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            context = V.cumsum(dim=-2)
        return context
    
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)
    
    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)
        
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # Add scale factor
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        # Get context
        context = self._get_initial_context(values, L_Q)
        
        # Update context
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class AttentionLayer(nn.Module):
    """
    Attention layer wrapper.
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix
    
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        
        return self.out_projection(out), attn

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    """
    Informer encoder layer.
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        
        return self.norm2(x+y), attn

class Encoder(nn.Module):
    """
    Informer encoder.
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
    
    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

class DecoderLayer(nn.Module):
    """
    Informer decoder layer.
    """
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)
        
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        
        return self.norm3(x+y)

class Decoder(nn.Module):
    """
    Informer decoder.
    """
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
    
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x

class DataEmbedding(nn.Module):
    """
    Data embedding layer.
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class InformerModel(nn.Module):
    """
    Informer model for long-term time series forecasting.
    """
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                 factor=5, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=512, 
                 dropout=0.0, attn='prob', activation='gelu', 
                 output_attention=False, distil=True, mix=True, device=torch.device('cuda:0')):
        super(InformerModel, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        
        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(ProbAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(ProbAttention(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(ProbAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:]

# Custom dataset for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_col='close'):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # Get sequence
        seq = self.data[idx:idx+self.seq_len].values
        
        # Get target
        target_idx = self.data.columns.get_loc(self.target_col)
        target = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len].iloc[:, target_idx].values
        
        return torch.FloatTensor(seq), torch.FloatTensor(target)

class InformerWrapper:
    """
    Wrapper class for the Informer model for stock price prediction.
    """
    def __init__(self, ticker: str, timeframe: str):
        """
        Initialize the Informer model wrapper.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Prediction timeframe (e.g., 'intraday', 'short_term', 'medium_term', 'long_term')
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.model = None
        self.feature_columns = []
        self.target_column = 'close'
        self.model_params = MODEL_PARAMS['informer'][timeframe]
        self.model_path = os.path.join('models', f'informer_{ticker}_{timeframe}.pt')
        self.scaler_path = os.path.join('models', f'informer_scaler_{ticker}_{timeframe}.joblib')
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set hyperparameters
        self.seq_len = self.model_params.get('seq_len', 60)  # Default: 60 days of historical data
        self.pred_len = self.model_params.get('pred_len', 5)  # Default: 5 days of prediction
        self.batch_size = self.model_params.get('batch_size', 32)
        self.learning_rate = self.model_params.get('learning_rate', 0.001)
        self.epochs = self.model_params.get('epochs', 50)
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for training or prediction.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                logger.error(f"Feature column {col} not found in data")
                return pd.DataFrame()
        
        # Select only the required columns
        df = df[self.feature_columns + [self.target_column] if self.target_column not in self.feature_columns else self.feature_columns]
        
        return df
    
    def train(self, data: pd.DataFrame, feature_columns: List[str]) -> None:
        """
        Train the Informer model.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            feature_columns: List of column names to use as features
        """
        self.feature_columns = feature_columns
        df = self.preprocess_data(data)
        
        if df.empty:
            logger.error(f"No valid data for training Informer model for {self.ticker}")
            return
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(df, self.seq_len, self.pred_len, self.target_column)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_dim = len(self.feature_columns)
        output_dim = 1  # Predicting only the target column
        self.model = InformerModel(input_dim, output_dim, self.seq_len, self.pred_len).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Train the model
        logger.info(f"Training Informer model for {self.ticker} with {len(dataset)} samples")
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Save the model
        self.save_model()
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            
        Returns:
            Array of predicted values for the prediction window
        """
        if self.model is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                logger.error(f"No trained model found for {self.ticker}")
                return np.array([])
        
        df = self.preprocess_data(data)
        
        if df.empty or len(df) < self.seq_len:
            logger.error(f"Insufficient data for prediction with Informer model for {self.ticker}")
            return np.array([])
        
        # Get the most recent sequence for prediction
        recent_data = df.iloc[-self.seq_len:].values
        recent_data = torch.FloatTensor(recent_data).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(recent_data)
            prediction = prediction.squeeze().cpu().numpy()
        
        return prediction
    
    def save_model(self) -> None:
        """
        Save the trained model to disk.
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_dim': self.model.input_dim,
                'output_dim': self.model.output_dim,
                'seq_len': self.seq_len,
                'pred_len': self.pred_len
            }, self.model_path)
            
            # Save additional metadata
            joblib.dump({
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'timeframe': self.timeframe,
                'ticker': self.ticker
            }, self.scaler_path)
            
            logger.info(f"Saved Informer model to {self.model_path}")
    
    def load_model(self) -> None:
        """
        Load a trained model from disk.
        """
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            # Load model metadata
            metadata = joblib.load(self.scaler_path)
            self.feature_columns = metadata['feature_columns']
            self.target_column = metadata['target_column']
            
            # Load model state
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model with the same architecture
            input_dim = checkpoint['input_dim']
            output_dim = checkpoint['output_dim']
            self.seq_len = checkpoint['seq_len']
            self.pred_len = checkpoint['pred_len']
            
            self.model = InformerModel(input_dim, output_dim, self.seq_len, self.pred_len).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info(f"Loaded Informer model from {self.model_path}")
        else:
            logger.error(f"No model file found at {self.model_path} or {self.scaler_path}")