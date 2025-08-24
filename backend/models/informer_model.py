import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta

from backend.utils.config import MODEL_PARAMS

logger = logging.getLogger(__name__)

# Placeholder for the Informer model architecture
# In a real implementation, you would import the actual Informer model from a library or implement it
class InformerModel(nn.Module):
    """
    Placeholder for the Informer model architecture.
    In a real implementation, this would be the actual Informer model.
    """
    def __init__(self, input_dim, output_dim, seq_len, pred_len):
        super(InformerModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Placeholder layers - in a real implementation, this would be the actual Informer architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * output_dim)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)
        
        # Reshape for encoder
        x = x.view(batch_size, -1)
        
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        # Reshape output to [batch_size, pred_len, output_dim]
        output = decoded.view(batch_size, self.pred_len, self.output_dim)
        
        return output

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