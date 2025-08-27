import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler

from backend.utils.config import MODEL_PARAMS

logger = logging.getLogger(__name__)

# Define the Dueling DQN network architecture
class DuelingDQNNetwork(nn.Module):
    """
    Dueling Deep Q-Network for trading decisions with separate value and advantage streams.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """
        Initialize the Dueling DQN network.
        
        Args:
            input_dim: Dimension of the input features
            output_dim: Dimension of the output (number of actions)
            hidden_dim: Hidden layer dimension
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Shared feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream - estimates state value V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Advantage stream - estimates action advantage A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the dueling network.
        
        Args:
            x: Input tensor
            
        Returns:
            Q-values for each action computed as Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        """
        # Extract features
        features = self.feature_layer(x)
        
        # Compute value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

# Define a namedtuple for storing experiences in the replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning.
    """
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling step
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priorities
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done, td_error: float = None):
        """
        Add an experience to the buffer with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            td_error: Temporal difference error for priority calculation
        """
        experience = Experience(state, action, reward, next_state, done)
        
        # Calculate priority based on TD error or use max priority for new experiences
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            priority = (np.max(self.priorities) if self.size > 0 else 1.0) ** self.alpha
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, weights)
        """
        if self.size == 0:
            return None, None, None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=True)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Extract experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        weights = torch.FloatTensor(weights)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return self.size

class DQNAgent:
    """
    Enhanced DQN agent with Dueling DQN, Double DQN, and Prioritized Experience Replay.
    """
    def __init__(self, ticker: str, timeframe: str):
        """
        Initialize the enhanced DQN agent.
        
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
        self.target_update_freq = self.model_params.get('target_update_freq', 100)
        self.double_dqn = self.model_params.get('double_dqn', True)
        self.prioritized_replay = self.model_params.get('prioritized_replay', True)
        
        # Initialize networks and optimizer
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
        # Initialize replay buffer
        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                self.buffer_size,
                alpha=self.model_params.get('alpha', 0.6),
                beta=self.model_params.get('beta', 0.4),
                beta_increment=self.model_params.get('beta_increment', 0.001)
            )
        else:
            # Fallback to simple replay buffer
            from collections import deque
            class SimpleReplayBuffer:
                def __init__(self, capacity):
                    self.buffer = deque(maxlen=capacity)
                def add(self, *args, **kwargs):
                    self.buffer.append(Experience(*args[:5]))
                def sample(self, batch_size):
                    experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
                    states = torch.FloatTensor([e.state for e in experiences])
                    actions = torch.LongTensor([e.action for e in experiences])
                    rewards = torch.FloatTensor([e.reward for e in experiences])
                    next_states = torch.FloatTensor([e.next_state for e in experiences])
                    dones = torch.FloatTensor([e.done for e in experiences])
                    return (states, actions, rewards, next_states, dones), None, None
                def __len__(self):
                    return len(self.buffer)
            self.memory = SimpleReplayBuffer(self.buffer_size)
        
        # Training counters
        self.step_count = 0
        self.episode_count = 0
        
        # Performance tracking
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon_values': [],
            'q_values': [],
            'td_errors': []
        }
        
        # Initialize feature columns and scaler
        self.feature_columns = []
        self.scaler = StandardScaler()
    
    def _initialize_networks(self, input_dim: int):
        """
        Initialize the policy and target networks using Dueling DQN architecture.
        
        Args:
            input_dim: Dimension of the input features
        """
        self.input_dim = input_dim
        hidden_dim = self.model_params.get('hidden_dim', 256)
        
        self.policy_net = DuelingDQNNetwork(input_dim, self.output_dim, hidden_dim).to(self.device)
        self.target_net = DuelingDQNNetwork(input_dim, self.output_dim, hidden_dim).to(self.device)
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
    
    def train(self, data: pd.DataFrame, feature_columns: List[str], episodes: int = 1000, 
              save_freq: int = 100, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the enhanced DQN agent.
        
        Args:
            data: DataFrame containing stock data with technical indicators and news sentiment
            feature_columns: List of column names to use as features
            episodes: Number of episodes to train for
            save_freq: Frequency of model saving
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history and metrics
        """
        try:
            self.feature_columns = feature_columns
            
            logger.info(f"Starting enhanced DQN training for {self.ticker} ({self.timeframe})")
            
            # Initialize networks if not already initialized
            if self.policy_net is None:
                self._initialize_networks(len(feature_columns))
            
            # Preprocess data
            states = self.preprocess_state(data)
            
            if len(states) == 0:
                logger.error(f"No valid data for training DQN agent for {self.ticker}")
                return {'error': 'No valid data'}
            
            # Split data for training and validation
            split_idx = int(len(states) * (1 - validation_split))
            train_states = states[:split_idx]
            val_states = states[split_idx:]
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]
            
            logger.info(f"Training DQN agent for {self.ticker} with {len(train_states)} training samples")
            
            # Training loop with early stopping
            best_val_reward = float('-inf')
            patience_counter = 0
            patience = 50
            
            for episode in range(episodes):
                # Training phase
                train_reward = self._run_episode(train_states, train_data, training=True)
                self.training_history['episode_rewards'].append(train_reward)
                self.training_history['epsilon_values'].append(self.epsilon)
                
                # Validation phase (every 10 episodes)
                if episode % 10 == 0:
                    val_reward = self._run_episode(val_states, val_data, training=False)
                    
                    # Early stopping check
                    if val_reward > best_val_reward:
                        best_val_reward = val_reward
                        patience_counter = 0
                        self.save_model()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at episode {episode}")
                        break
                
                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # Update target network periodically
                if episode % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Log progress
                if (episode + 1) % 100 == 0:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    avg_loss = np.mean(self.training_history['losses'][-100:]) if self.training_history['losses'] else 0
                    logger.info(f"Episode {episode+1}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                              f"Avg Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
                
                # Save model periodically
                if episode % save_freq == 0 and episode > 0:
                    self.save_model()
            
            # Final model save
            self.save_model()
            
            # Calculate comprehensive training metrics
            final_metrics = {
                'total_episodes': len(self.training_history['episode_rewards']),
                'final_epsilon': self.epsilon,
                'avg_reward': np.mean(self.training_history['episode_rewards']),
                'max_reward': np.max(self.training_history['episode_rewards']),
                'min_reward': np.min(self.training_history['episode_rewards']),
                'reward_std': np.std(self.training_history['episode_rewards']),
                'best_val_reward': best_val_reward,
                'avg_loss': np.mean(self.training_history['losses']) if self.training_history['losses'] else 0,
                'avg_td_error': np.mean(self.training_history['td_errors']) if self.training_history['td_errors'] else 0,
                'training_history': self.training_history
            }
            
            logger.info(f"Enhanced DQN training completed. Final metrics: {final_metrics}")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error during DQN training: {str(e)}")
            raise
    
    def _run_episode(self, states: np.ndarray, data: pd.DataFrame, training: bool = True) -> float:
        """
        Run a single episode of trading simulation.
        
        Args:
            states: Processed feature states
            data: Original price data
            training: Whether this is a training episode
            
        Returns:
            Total episode reward
        """
        episode_reward = 0
        portfolio_value = 10000  # Starting portfolio value
        cash = portfolio_value
        shares = 0
        transaction_cost = 0.001  # 0.1% transaction cost
        episode_length = 0
        
        for i in range(len(states) - 1):
            state = states[i]
            next_state = states[i + 1]
            
            # Select action
            if training:
                action = self.select_action(state, training=True)
            else:
                # Use greedy policy for validation
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
            
            # Execute action and calculate reward
            current_price = data.iloc[i]['close']
            next_price = data.iloc[i + 1]['close']
            
            reward = self._calculate_reward(action, current_price, next_price, 
                                          cash, shares, transaction_cost)
            
            # Update portfolio based on action
            if action == 0 and cash > current_price:  # Buy
                shares_to_buy = int(cash / current_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + transaction_cost)
                    if cost <= cash:
                        cash -= cost
                        shares += shares_to_buy
            elif action == 2 and shares > 0:  # Sell
                revenue = shares * current_price * (1 - transaction_cost)
                cash += revenue
                shares = 0
            # Hold: no action needed
            
            # Store experience for training
            if training:
                done = (i == len(states) - 2)
                # Calculate TD error for prioritized replay
                td_error = None
                if hasattr(self, 'policy_net') and len(self.memory) > 0:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                        current_q = self.policy_net(state_tensor)
                        next_q = self.target_net(next_state_tensor)
                        target_q = reward + self.gamma * next_q.max().item() * (1 - done)
                        td_error = abs(current_q[0][action].item() - target_q)
                
                self.memory.add(state, action, reward, next_state, done, td_error)
                
                # Perform experience replay
                if len(self.memory) > self.batch_size:
                    self._experience_replay()
            
            episode_reward += reward
            episode_length += 1
        
        if training:
            self.training_history['episode_lengths'].append(episode_length)
        
        return episode_reward
    
    def _calculate_reward(self, action: int, current_price: float, next_price: float,
                        cash: float, shares: int, transaction_cost: float) -> float:
        """
        Calculate reward based on action and market movement.
        
        Args:
            action: Action taken (0: buy, 1: hold, 2: sell)
            current_price: Current stock price
            next_price: Next period stock price
            cash: Available cash
            shares: Current shares held
            transaction_cost: Transaction cost rate
            
        Returns:
            Calculated reward
        """
        price_change = (next_price - current_price) / current_price
        
        # Base reward components
        directional_reward = 0
        opportunity_cost = 0
        transaction_penalty = 0
        
        if action == 0:  # Buy
            if cash > current_price:  # Valid buy
                directional_reward = price_change * 100  # Reward for correct direction
                transaction_penalty = -transaction_cost * 10  # Small transaction cost
            else:
                directional_reward = -5  # Penalty for invalid action
        
        elif action == 2:  # Sell
            if shares > 0:  # Valid sell
                directional_reward = -price_change * 100  # Reward for correct direction
                transaction_penalty = -transaction_cost * 10  # Small transaction cost
            else:
                directional_reward = -5  # Penalty for invalid action
        
        else:  # Hold
            # Small penalty for missing significant moves
            if abs(price_change) > 0.02:  # 2% threshold
                opportunity_cost = -abs(price_change) * 20
            else:
                directional_reward = 1  # Small reward for avoiding unnecessary trades
        
        # Risk-adjusted reward
        volatility_penalty = -abs(price_change) * 5 if abs(price_change) > 0.05 else 0
        
        total_reward = directional_reward + opportunity_cost + transaction_penalty + volatility_penalty
        
        return total_reward
    
    def _experience_replay(self):
        """
        Perform experience replay with Double DQN and prioritized sampling.
        """
        # Sample a batch of experiences
        batch_data, indices, weights = self.memory.sample(self.batch_size)
        
        if batch_data is None:
            return
        
        states, actions, rewards, next_states, dones = batch_data
        
        # Move tensors to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q-values for current states and actions
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use policy network to select actions, target network to evaluate
        if self.double_dqn:
            # Select actions using policy network
            next_actions = self.policy_net(next_states).max(1)[1].detach()
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
        else:
            # Standard DQN: Use target network for both selection and evaluation
            next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        # Compute target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute TD errors for priority updates
        td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
        
        # Compute loss with importance sampling weights if using prioritized replay
        if self.prioritized_replay and weights is not None:
            weights = weights.to(self.device)
            loss = (weights * nn.MSELoss(reduction='none')(current_q_values, target_q_values)).mean()
        else:
            loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors)
        
        # Track training metrics
        self.training_history['losses'].append(loss.item())
        self.training_history['td_errors'].append(np.mean(np.abs(td_errors)))
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
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
    
    def get_trading_signal(self, data: pd.DataFrame, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Get enhanced trading signal with confidence intervals and risk assessment.
        
        Args:
            data: Latest market data
            return_probabilities: Whether to return action probabilities
            
        Returns:
            Dictionary containing trading signal, confidence, and additional metrics
        """
        try:
            # Preprocess the data
            states = self.preprocess_state(data)
            
            if len(states) == 0:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'q_values': [0.0, 0.0, 0.0],
                    'error': 'No valid data for prediction'
                }
            
            # Get the latest state
            latest_state = states[-1]
            
            # Get Q-values from both policy and target networks for ensemble
            with torch.no_grad():
                state_tensor = torch.FloatTensor(latest_state).unsqueeze(0).to(self.device)
                
                # Policy network Q-values
                policy_q_values = self.policy_net(state_tensor)
                policy_q_np = policy_q_values.cpu().numpy()[0]
                
                # Target network Q-values for stability check
                target_q_values = self.target_net(state_tensor)
                target_q_np = target_q_values.cpu().numpy()[0]
                
                # Ensemble Q-values (weighted average)
                ensemble_q = 0.7 * policy_q_np + 0.3 * target_q_np
            
            # Select action (greedy policy for inference)
            action = np.argmax(ensemble_q)
            
            # Convert action to signal
            signal_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
            signal = signal_map[action]
            
            # Enhanced confidence calculation
            max_q = np.max(ensemble_q)
            second_max_q = np.partition(ensemble_q, -2)[-2]
            confidence = (max_q - second_max_q) / (np.abs(max_q) + 1e-8)
            confidence = min(max(confidence, 0.0), 1.0)
            
            # Risk assessment
            q_std = np.std(ensemble_q)
            risk_level = 'low' if q_std < 0.5 else 'medium' if q_std < 1.0 else 'high'
            
            # Market regime detection (simplified)
            recent_volatility = np.std(data['close'].pct_change().dropna().tail(20))
            market_regime = 'volatile' if recent_volatility > 0.03 else 'stable'
            
            result = {
                'signal': signal,
                'confidence': float(confidence),
                'q_values': ensemble_q.tolist(),
                'risk_level': risk_level,
                'market_regime': market_regime,
                'volatility': float(recent_volatility),
                'model_agreement': float(np.corrcoef(policy_q_np, target_q_np)[0, 1])
            }
            
            if return_probabilities:
                # Convert Q-values to probabilities using softmax
                exp_q = np.exp(ensemble_q - np.max(ensemble_q))
                probabilities = exp_q / np.sum(exp_q)
                
                result['action_probabilities'] = {
                    'buy': float(probabilities[0]),
                    'hold': float(probabilities[1]),
                    'sell': float(probabilities[2])
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting trading signal: {str(e)}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'q_values': [0.0, 0.0, 0.0],
                'error': str(e)
            }
    
    def evaluate(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Evaluate the DQN model performance on test data.
        
        Args:
            data: Test data
            feature_columns: Feature columns to use
            
        Returns:
            Evaluation metrics
        """
        try:
            self.feature_columns = feature_columns
            states = self.preprocess_state(data)
            
            if len(states) == 0:
                return {'error': 'No valid data for evaluation'}
            
            # Run evaluation episode
            total_reward = self._run_episode(states, data, training=False)
            
            # Calculate additional metrics
            signals = []
            confidences = []
            
            for i in range(len(states)):
                signal_data = self.get_trading_signal(data.iloc[i:i+1])
                signals.append(signal_data['signal'])
                confidences.append(signal_data['confidence'])
            
            # Trading performance metrics
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            hold_signals = signals.count('HOLD')
            
            avg_confidence = np.mean(confidences)
            
            # Price prediction accuracy (simplified)
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(len(data) - 1):
                current_price = data.iloc[i]['close']
                next_price = data.iloc[i + 1]['close']
                price_change = next_price > current_price
                
                if signals[i] == 'BUY' and price_change:
                    correct_predictions += 1
                elif signals[i] == 'SELL' and not price_change:
                    correct_predictions += 1
                elif signals[i] == 'HOLD':
                    correct_predictions += 0.5  # Neutral for hold
                
                total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            return {
                'total_reward': total_reward,
                'avg_confidence': avg_confidence,
                'signal_distribution': {
                    'buy': buy_signals,
                    'hold': hold_signals,
                    'sell': sell_signals
                },
                'prediction_accuracy': accuracy,
                'total_episodes_trained': len(self.training_history['episode_rewards']),
                'final_epsilon': self.epsilon
            }
            
        except Exception as e:
            logger.error(f"Error during DQN evaluation: {str(e)}")
            return {'error': str(e)}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary and statistics.
        
        Returns:
            Model summary dictionary
        """
        try:
            summary = {
                'model_type': 'Enhanced DQN with Dueling Architecture',
                'ticker': self.ticker,
                'timeframe': self.timeframe,
                'architecture': {
                    'network_type': 'Dueling DQN',
                    'double_dqn': self.double_dqn,
                    'prioritized_replay': self.prioritized_replay,
                    'input_dim': getattr(self, 'input_dim', 'Not initialized'),
                    'output_dim': self.output_dim,
                    'hidden_dim': self.model_params.get('hidden_dim', 256)
                },
                'hyperparameters': {
                    'learning_rate': self.lr,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon,
                    'epsilon_min': self.epsilon_min,
                    'epsilon_decay': self.epsilon_decay,
                    'batch_size': self.batch_size,
                    'buffer_size': self.buffer_size,
                    'target_update_freq': self.target_update_freq
                },
                'training_status': {
                    'is_trained': hasattr(self, 'policy_net') and self.policy_net is not None,
                    'total_episodes': len(self.training_history['episode_rewards']),
                    'memory_size': len(self.memory) if hasattr(self, 'memory') else 0
                }
            }
            
            # Add training history if available
            if self.training_history['episode_rewards']:
                summary['training_metrics'] = {
                    'avg_reward': np.mean(self.training_history['episode_rewards']),
                    'max_reward': np.max(self.training_history['episode_rewards']),
                    'min_reward': np.min(self.training_history['episode_rewards']),
                    'reward_trend': 'improving' if len(self.training_history['episode_rewards']) > 10 and 
                                   np.mean(self.training_history['episode_rewards'][-10:]) > 
                                   np.mean(self.training_history['episode_rewards'][:10]) else 'stable'
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return {'error': str(e)}
    
    def reset_model(self) -> None:
        """
        Reset the model to untrained state.
        """
        try:
            # Reset networks
            if hasattr(self, 'policy_net') and self.policy_net is not None:
                self.policy_net = None
                self.target_net = None
                self.optimizer = None
            
            # Reset training state
            self.epsilon = 1.0
            self.step_count = 0
            self.episode_count = 0
            
            # Clear training history
            self.training_history = {
                'episode_rewards': [],
                'episode_lengths': [],
                'losses': [],
                'epsilon_values': [],
                'q_values': [],
                'td_errors': []
            }
            
            # Reset memory buffer
            if hasattr(self, 'memory'):
                if self.prioritized_replay:
                    self.memory = PrioritizedReplayBuffer(
                        self.buffer_size,
                        alpha=self.model_params.get('alpha', 0.6),
                        beta=self.model_params.get('beta', 0.4),
                        beta_increment=self.model_params.get('beta_increment', 0.001)
                    )
                else:
                    from collections import deque
                    class SimpleReplayBuffer:
                        def __init__(self, capacity):
                            self.buffer = deque(maxlen=capacity)
                        def add(self, *args, **kwargs):
                            self.buffer.append(Experience(*args[:5]))
                        def sample(self, batch_size):
                            experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
                            states = torch.FloatTensor([e.state for e in experiences])
                            actions = torch.LongTensor([e.action for e in experiences])
                            rewards = torch.FloatTensor([e.reward for e in experiences])
                            next_states = torch.FloatTensor([e.next_state for e in experiences])
                            dones = torch.FloatTensor([e.done for e in experiences])
                            return (states, actions, rewards, next_states, dones), None, None
                        def __len__(self):
                            return len(self.buffer)
                    self.memory = SimpleReplayBuffer(self.buffer_size)
            
            logger.info(f"DQN model reset for {self.ticker} ({self.timeframe})")
            
        except Exception as e:
            logger.error(f"Error resetting DQN model: {str(e)}")
            raise