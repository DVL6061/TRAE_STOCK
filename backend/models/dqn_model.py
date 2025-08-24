import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime, timedelta

from backend.utils.config import MODEL_PARAMS

logger = logging.getLogger(__name__)

# Define the DQN network architecture
class DQNNetwork(nn.Module):
    """
    Deep Q-Network for trading decisions.
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Initialize the DQN network.
        
        Args:
            input_dim: Dimension of the input features
            output_dim: Dimension of the output (number of actions)
        """
        super(DQNNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action
        """
        return self.layers(x)

# Define a namedtuple for storing experiences in the replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    """
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of experiences
        """
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    DQN agent for trading decisions.
    """
    def __init__(self, ticker: str, timeframe: str):
        """
        Initialize the DQN agent.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Trading timeframe (e.g., 'intraday', 'short_term', 'medium_term', 'long_term')
        """
        self.ticker = ticker
        self.timeframe = timeframe
        self.model_params = MODEL_PARAMS['dqn'][timeframe]
        self.model_path = os.path.join('models', f'dqn_{ticker}_{timeframe}.pt')
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set hyperparameters
        self.input_dim = 0  # Will be set during training
        self.output_dim = 3  # Buy (0), Hold (1), Sell (2)
        self.gamma = self.model_params.get('gamma', 0.99)  # Discount factor
        self.epsilon = self.model_params.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = self.model_params.get('epsilon_min', 0.01)
        self.epsilon_decay = self.model_params.get('epsilon_decay', 0.995)
        self.learning_rate = self.model_params.get('learning_rate', 0.001)
        self.batch_size = self.model_params.get('batch_size', 64)
        self.buffer_size = self.model_params.get('buffer_size', 10000)
        
        # Initialize networks and optimizer
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(self.buffer_size)
        
        # Initialize feature columns
        self.feature_columns = []
    
    def _initialize_networks(self, input_dim: int):
        """
        Initialize the policy and target networks.
        
        Args:
            input_dim: Dimension of the input features
        """
        self.input_dim = input_dim
        self.policy_net = DQNNetwork(input_dim, self.output_dim).to(self.device)
        self.target_net = DQNNetwork(input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def preprocess_state(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data to create a state representation.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            
        Returns:
            State representation as a numpy array
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Handle missing values
        df = df.dropna()
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                logger.error(f"Feature column {col} not found in data")
                return np.array([])
        
        # Extract features
        state = df[self.feature_columns].values
        
        return state
    
    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            training: Whether the agent is in training mode
            
        Returns:
            Selected action (0: Buy, 1: Hold, 2: Sell)
        """
        if self.policy_net is None:
            if os.path.exists(self.model_path):
                self.load_model()
            else:
                logger.error(f"No trained model found for {self.ticker}")
                return 1  # Default to Hold
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection during training
        if training and random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        
        # Select action with highest Q-value
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def train(self, data: pd.DataFrame, feature_columns: List[str], episodes: int = 100) -> None:
        """
        Train the DQN agent.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            feature_columns: List of column names to use as features
            episodes: Number of episodes to train for
        """
        self.feature_columns = feature_columns
        
        # Initialize networks if not already initialized
        if self.policy_net is None:
            self._initialize_networks(len(feature_columns))
        
        # Preprocess data
        states = self.preprocess_state(data)
        
        if len(states) == 0:
            logger.error(f"No valid data for training DQN agent for {self.ticker}")
            return
        
        logger.info(f"Training DQN agent for {self.ticker} with {len(states)} samples")
        
        # Training loop
        for episode in range(episodes):
            total_reward = 0
            
            # Reset state to beginning of data
            current_idx = 0
            state = states[current_idx]
            done = False
            
            while not done:
                # Select action
                action = self.select_action(state, training=True)
                
                # Move to next state
                current_idx += 1
                if current_idx >= len(states):
                    next_state = state  # Use current state as next state if at end of data
                    done = True
                else:
                    next_state = states[current_idx]
                
                # Calculate reward (simplified for this implementation)
                # In a real implementation, this would be based on profit/loss
                if action == 0:  # Buy
                    reward = 1 if current_idx < len(states) - 1 and data.iloc[current_idx+1]['close'] > data.iloc[current_idx]['close'] else -1
                elif action == 2:  # Sell
                    reward = 1 if current_idx < len(states) - 1 and data.iloc[current_idx+1]['close'] < data.iloc[current_idx]['close'] else -1
                else:  # Hold
                    reward = 0.1  # Small positive reward for holding
                
                # Store experience in replay buffer
                self.memory.add(state, action, reward, next_state, done)
                
                # Update state and total reward
                state = next_state
                total_reward += reward
                
                # Perform experience replay if enough samples are available
                if len(self.memory) >= self.batch_size:
                    self._experience_replay()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network periodically
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Log progress
            if (episode + 1) % 10 == 0:
                logger.info(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        # Save the model
        self.save_model()
    
    def _experience_replay(self):
        """
        Perform experience replay to update the policy network.
        """
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q-values for current states and actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next state values using target network
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self) -> None:
        """
        Save the trained model to disk.
        """
        if self.policy_net is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'input_dim': self.input_dim,
                'output_dim': self.output_dim,
                'feature_columns': self.feature_columns,
                'epsilon': self.epsilon
            }, self.model_path)
            
            logger.info(f"Saved DQN model to {self.model_path}")
    
    def load_model(self) -> None:
        """
        Load a trained model from disk.
        """
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Get model parameters
            self.input_dim = checkpoint['input_dim']
            self.output_dim = checkpoint['output_dim']
            self.feature_columns = checkpoint['feature_columns']
            self.epsilon = checkpoint['epsilon']
            
            # Initialize networks
            self._initialize_networks(self.input_dim)
            
            # Load model state
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Set target network to evaluation mode
            self.target_net.eval()
            
            logger.info(f"Loaded DQN model from {self.model_path}")
        else:
            logger.error(f"No model file found at {self.model_path}")
    
    def get_trading_signal(self, data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Get a trading signal based on the current state.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            
        Returns:
            Dictionary with trading signal and confidence
        """
        # Preprocess data to get current state
        states = self.preprocess_state(data)
        
        if len(states) == 0:
            logger.error(f"No valid data for generating trading signal for {self.ticker}")
            return {'signal': 'HOLD', 'confidence': 0.0}
        
        # Get current state (last row)
        current_state = states[-1]
        
        # Select action
        action = self.select_action(current_state)
        
        # Convert action to signal
        signal_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
        signal = signal_map[action]
        
        # Calculate confidence (simplified for this implementation)
        # In a real implementation, this would be based on Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)
            max_q_value = q_values.max().item()
            min_q_value = q_values.min().item()
            range_q_value = max(1e-5, max_q_value - min_q_value)  # Avoid division by zero
            confidence = (q_values[action].item() - min_q_value) / range_q_value
        
        return {'signal': signal, 'confidence': confidence}