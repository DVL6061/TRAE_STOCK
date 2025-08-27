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
    def __init__(self, x_enc, x_dec, y):
        """
        Dataset for time series data.
        
        Args:
            x_enc: Encoder input sequences
            x_dec: Decoder input sequences  
            y: Target sequences
        """
        self.x_enc = torch.FloatTensor(x_enc)
        self.x_dec = torch.FloatTensor(x_dec)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.x_enc)
    
    def __getitem__(self, idx):
        return self.x_enc[idx], self.x_dec[idx], self.y[idx]

class InformerWrapper:
    """
    Enhanced wrapper class for the Informer model for stock price prediction.
    """
    def __init__(self, ticker: str = None, timeframe: str = 'daily', seq_len: int = 96, pred_len: int = 24, 
                 enc_in: int = 7, dec_in: int = 7, c_out: int = 1, d_model: int = 512, 
                 n_heads: int = 8, e_layers: int = 2, d_layers: int = 1, d_ff: int = 2048, 
                 factor: int = 5, dropout: float = 0.05, attn: str = 'prob', 
                 activation: str = 'gelu', output_attention: bool = False, 
                 distil: bool = True, mix: bool = True):
        """
        Initialize the Informer model wrapper.
        
        Args:
            ticker: Stock ticker symbol (optional for generic use)
            timeframe: Prediction timeframe
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            enc_in: Encoder input size
            dec_in: Decoder input size
            c_out: Output size
            d_model: Model dimension
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_layers: Number of decoder layers
            d_ff: Feed forward dimension
            factor: Attention factor
            dropout: Dropout rate
            attn: Attention type
            activation: Activation function
            output_attention: Whether to output attention
            distil: Whether to use distillation
            mix: Whether to use mix attention
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = seq_len // 2  # Half of sequence length for decoder input
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        
        # Model parameters
        self.model_params = {
            'd_model': d_model,
            'n_heads': n_heads,
            'e_layers': e_layers,
            'd_layers': d_layers,
            'd_ff': d_ff,
            'factor': factor,
            'dropout': dropout,
            'attn': attn,
            'activation': activation,
            'output_attention': output_attention,
            'distil': distil,
            'mix': mix
        }
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = {}
        self.model_metrics = {}
        self.feature_columns = []
        self.target_column = 'close'
        
        # File paths
        if ticker:
            self.model_path = os.path.join('models', f'informer_{ticker}_{timeframe}.pt')
            self.scaler_path = os.path.join('models', f'informer_scaler_{ticker}_{timeframe}.joblib')
        else:
            self.model_path = os.path.join('models', f'informer_{timeframe}.pt')
            self.scaler_path = os.path.join('models', f'informer_scaler_{timeframe}.joblib')
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        logger.info(f"InformerWrapper initialized with device: {self.device}")
    
    def _create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            targets: Target array of shape (n_samples,)
            
        Returns:
            Tuple of (X, y) where X contains encoder and decoder inputs, y contains targets
        """
        X_enc, X_dec, y = [], [], []
        
        for i in range(len(features) - self.seq_len - self.pred_len + 1):
            # Encoder input: seq_len historical data
            enc_input = features[i:i+self.seq_len]
            
            # Decoder input: label_len + pred_len
            # First part: last label_len from encoder
            # Second part: zeros for prediction
            dec_input = np.zeros((self.label_len + self.pred_len, features.shape[1]))
            dec_input[:self.label_len] = features[i+self.seq_len-self.label_len:i+self.seq_len]
            
            # Target: next pred_len values
            target = targets[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            X_enc.append(enc_input)
            X_dec.append(dec_input)
            y.append(target)
        
        return np.array(X_enc), np.array(X_dec), np.array(y)
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocess data for the Informer model.
        
        Args:
            data: Raw stock data with OHLCV and features
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        try:
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Sort by date if date column exists
            if 'date' in df.columns:
                df = df.sort_values('date')
            elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
            
            # Remove non-numeric columns except target
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Ensure target column is included
            if self.target_column not in numeric_columns:
                if self.target_column in df.columns:
                    df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
                    numeric_columns.append(self.target_column)
                else:
                    raise ValueError(f"Target column '{self.target_column}' not found in data")
            
            # Select numeric features
            df_numeric = df[numeric_columns].copy()
            
            # Handle missing values
            df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
            
            # Store feature columns (excluding target)
            self.feature_columns = [col for col in numeric_columns if col != self.target_column]
            
            # Prepare features and target
            features = df_numeric[self.feature_columns].values
            targets = df_numeric[self.target_column].values.reshape(-1, 1)
            
            # Scale features and targets
            features_scaled = self.scaler.fit_transform(features)
            targets_scaled = self.scaler.fit_transform(targets)
            
            logger.info(f"Data preprocessed: {features_scaled.shape[0]} samples, {features_scaled.shape[1]} features")
            
            return features_scaled, targets_scaled.flatten(), self.feature_columns
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def train(self, data: pd.DataFrame, feature_columns: List[str] = None, 
              validation_split: float = 0.2, epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.0001, patience: int = 10, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the Informer model with validation and early stopping.
        
        Args:
            data: Training data DataFrame
            feature_columns: List of feature column names (optional)
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history and metrics
        """
        try:
            logger.info("Starting Informer model training...")
            
            # Preprocess data
            features, targets, feature_names = self.preprocess_data(data)
            
            if feature_columns:
                self.feature_columns = feature_columns
            else:
                self.feature_columns = feature_names
            
            # Update input dimensions based on actual data
            self.enc_in = len(self.feature_columns)
            self.dec_in = len(self.feature_columns)
            
            # Create sequences for time series prediction
            X_enc, X_dec, y = self._create_sequences(features, targets)
            
            # Split data into train and validation
            split_idx = int(len(X_enc) * (1 - validation_split))
            X_enc_train, X_enc_val = X_enc[:split_idx], X_enc[split_idx:]
            X_dec_train, X_dec_val = X_dec[:split_idx], X_dec[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Create data loaders
            train_dataset = TimeSeriesDataset(X_enc_train, X_dec_train, y_train)
            val_dataset = TimeSeriesDataset(X_enc_val, X_dec_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model
            self.model = InformerModel(
                enc_in=self.enc_in,
                dec_in=self.dec_in,
                c_out=self.c_out,
                seq_len=self.seq_len,
                label_len=self.label_len,
                out_len=self.pred_len,
                **self.model_params
            ).to(self.device)
            
            # Initialize optimizer and loss function
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2, factor=0.5)
            
            # Training history
            history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': []
            }
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_losses = []
                
                for batch_x_enc, batch_x_dec, batch_y in train_loader:
                    batch_x_enc = batch_x_enc.to(self.device)
                    batch_x_dec = batch_x_dec.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_x_enc, batch_x_dec)
                    
                    # Calculate loss on prediction part only
                    loss = criterion(outputs[:, -self.pred_len:, :], batch_y.unsqueeze(-1))
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # Validation phase
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_x_enc, batch_x_dec, batch_y in val_loader:
                        batch_x_enc = batch_x_enc.to(self.device)
                        batch_x_dec = batch_x_dec.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x_enc, batch_x_dec)
                        loss = criterion(outputs[:, -self.pred_len:, :], batch_y.unsqueeze(-1))
                        val_losses.append(loss.item())
                
                # Calculate average losses
                avg_train_loss = np.mean(train_losses)
                avg_val_loss = np.mean(val_losses)
                current_lr = optimizer.param_groups[0]['lr']
                
                # Update history
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['learning_rate'].append(current_lr)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_val_loss
                    }, self.model_path.replace('.pt', '_best.pt'))
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.8f}")
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            if os.path.exists(self.model_path.replace('.pt', '_best.pt')):
                checkpoint = torch.load(self.model_path.replace('.pt', '_best.pt'), map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Calculate final metrics
            self.model.eval()
            with torch.no_grad():
                train_predictions = []
                train_targets = []
                
                for batch_x_enc, batch_x_dec, batch_y in train_loader:
                    batch_x_enc = batch_x_enc.to(self.device)
                    batch_x_dec = batch_x_dec.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x_enc, batch_x_dec)
                    
                    train_predictions.extend(outputs[:, -self.pred_len:, :].cpu().numpy().flatten())
                    train_targets.extend(batch_y.cpu().numpy().flatten())
                
                # Calculate metrics
                train_predictions = np.array(train_predictions)
                train_targets = np.array(train_targets)
                
                rmse = np.sqrt(mean_squared_error(train_targets, train_predictions))
                mae = mean_absolute_error(train_targets, train_predictions)
                r2 = r2_score(train_targets, train_predictions)
                
                self.model_metrics = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'final_train_loss': history['train_loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'best_val_loss': best_val_loss,
                    'epochs_trained': len(history['train_loss'])
                }
            
            self.training_history = history
            self.is_trained = True
            
            logger.info(f"Training completed - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
            
            return {
                'history': history,
                'metrics': self.model_metrics,
                'feature_columns': self.feature_columns
            }
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
    def predict(self, data: pd.DataFrame, return_confidence: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions using the trained Informer model.
        
        Args:
            data: DataFrame containing stock data with features
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Array of predicted values or dict with predictions and confidence intervals
        """
        try:
            if self.model is None:
                if os.path.exists(self.model_path):
                    self.load_model()
                else:
                    logger.error(f"No trained model found for {self.ticker or 'model'}")
                    return np.array([])
            
            # Preprocess data
            features, targets, _ = self.preprocess_data(data)
            
            if len(features) < self.seq_len:
                logger.error(f"Insufficient data for prediction. Need at least {self.seq_len} samples, got {len(features)}")
                return np.array([])
            
            # Get the most recent sequence for prediction
            recent_features = features[-self.seq_len:]
            recent_targets = targets[-self.seq_len:] if len(targets) >= self.seq_len else np.zeros(self.seq_len)
            
            # Create encoder input
            x_enc = recent_features.reshape(1, self.seq_len, -1)  # (1, seq_len, n_features)
            
            # Create decoder input
            x_dec = np.zeros((1, self.label_len + self.pred_len, features.shape[1]))
            # Use last label_len values from encoder for decoder initialization
            x_dec[0, :self.label_len] = recent_features[-self.label_len:]
            
            # Convert to tensors
            x_enc = torch.FloatTensor(x_enc).to(self.device)
            x_dec = torch.FloatTensor(x_dec).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Make prediction
            with torch.no_grad():
                if return_confidence:
                    # Multiple forward passes for confidence estimation
                    predictions = []
                    n_samples = 10
                    
                    for _ in range(n_samples):
                        # Add small noise for uncertainty estimation
                        x_enc_noisy = x_enc + torch.randn_like(x_enc) * 0.01
                        output = self.model(x_enc_noisy, x_dec)
                        pred = output[:, -self.pred_len:, :].cpu().numpy().flatten()
                        predictions.append(pred)
                    
                    predictions = np.array(predictions)
                    mean_pred = np.mean(predictions, axis=0)
                    std_pred = np.std(predictions, axis=0)
                    
                    # Calculate confidence intervals (95%)
                    confidence_lower = mean_pred - 1.96 * std_pred
                    confidence_upper = mean_pred + 1.96 * std_pred
                    
                    return {
                        'predictions': mean_pred,
                        'confidence_lower': confidence_lower,
                        'confidence_upper': confidence_upper,
                        'std': std_pred
                    }
                else:
                    output = self.model(x_enc, x_dec)
                    prediction = output[:, -self.pred_len:, :].cpu().numpy().flatten()
                    return prediction
                    
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return np.array([])
    
    def save_model(self) -> None:
        """
        Save the trained model and metadata to disk.
        """
        try:
            if self.model is not None:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                
                # Save model state
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_params': self.model_params,
                    'enc_in': self.enc_in,
                    'dec_in': self.dec_in,
                    'c_out': self.c_out,
                    'seq_len': self.seq_len,
                    'label_len': self.label_len,
                    'pred_len': self.pred_len,
                    'training_history': self.training_history,
                    'model_metrics': self.model_metrics,
                    'is_trained': self.is_trained,
                    'timestamp': datetime.now().isoformat()
                }, self.model_path)
                
                # Save scaler and metadata
                joblib.dump({
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'timeframe': self.timeframe,
                    'ticker': self.ticker,
                    'timestamp': datetime.now().isoformat()
                }, self.scaler_path)
                
                logger.info(f"Informer model saved to {self.model_path}")
            else:
                logger.error("No model to save")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self) -> None:
        """
        Load a trained model from disk.
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                # Load scaler and metadata
                metadata = joblib.load(self.scaler_path)
                self.scaler = metadata['scaler']
                self.feature_columns = metadata['feature_columns']
                self.target_column = metadata['target_column']
                self.timeframe = metadata.get('timeframe', self.timeframe)
                self.ticker = metadata.get('ticker', self.ticker)
                
                # Load model state
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Restore model parameters
                self.enc_in = checkpoint['enc_in']
                self.dec_in = checkpoint['dec_in']
                self.c_out = checkpoint['c_out']
                self.seq_len = checkpoint['seq_len']
                self.label_len = checkpoint['label_len']
                self.pred_len = checkpoint['pred_len']
                self.training_history = checkpoint.get('training_history', {})
                self.model_metrics = checkpoint.get('model_metrics', {})
                self.is_trained = checkpoint.get('is_trained', False)
                
                # Initialize model with the same architecture
                self.model = InformerModel(
                    enc_in=self.enc_in,
                    dec_in=self.dec_in,
                    c_out=self.c_out,
                    seq_len=self.seq_len,
                    label_len=self.label_len,
                    out_len=self.pred_len,
                    **self.model_params
                ).to(self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                logger.info(f"Loaded Informer model from {self.model_path}")
            else:
                logger.error(f"No model file found at {self.model_path} or {self.scaler_path}")
                
        except Exception as e:
             logger.error(f"Error loading model: {str(e)}")
             raise
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        """
        try:
            if not self.is_trained:
                logger.error("Model is not trained yet")
                return {}
            
            # Preprocess data
            features, targets, _ = self.preprocess_data(data)
            
            if len(features) == 0:
                logger.error("No valid data for evaluation")
                return {}
            
            # Create sequences
            x_enc, x_dec, y = self._create_sequences(features, targets)
            
            if len(x_enc) == 0:
                logger.error("Not enough data to create sequences")
                return {}
            
            # Convert to tensors
            x_enc = torch.FloatTensor(x_enc).to(self.device)
            x_dec = torch.FloatTensor(x_dec).to(self.device)
            y_true = torch.FloatTensor(y).to(self.device)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(x_enc, x_dec)
                y_pred = y_pred[:, -self.pred_len:, :]
            
            # Convert to numpy for metrics calculation
            y_true_np = y_true.cpu().numpy().flatten()
            y_pred_np = y_pred.cpu().numpy().flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_true_np, y_pred_np)
            mae = mean_absolute_error(y_true_np, y_pred_np)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_np, y_pred_np)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((y_true_np - y_pred_np) / (y_true_np + 1e-8))) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            logger.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture and training status.
        """
        try:
            summary = {
                'model_type': 'Informer',
                'ticker': self.ticker,
                'timeframe': self.timeframe,
                'architecture': {
                    'enc_in': self.enc_in,
                    'dec_in': self.dec_in,
                    'c_out': self.c_out,
                    'seq_len': self.seq_len,
                    'label_len': self.label_len,
                    'pred_len': self.pred_len,
                    **self.model_params
                },
                'is_trained': self.is_trained,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'training_history': self.training_history,
                'model_metrics': self.model_metrics,
                'device': str(self.device)
            }
            
            if self.model is not None:
                # Count parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                summary['parameters'] = {
                    'total': total_params,
                    'trainable': trainable_params
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating model summary: {str(e)}")
            return {}
    
    def reset_model(self) -> None:
        """
        Reset the model to untrained state.
        """
        try:
            self.model = None
            self.scaler = StandardScaler()
            self.feature_columns = []
            self.target_column = None
            self.training_history = {}
            self.model_metrics = {}
            self.is_trained = False
            
            logger.info("Model reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting model: {str(e)}")
            raise