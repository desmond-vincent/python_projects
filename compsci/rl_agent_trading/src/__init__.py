"""
TradeMaster - Deep Q-Network Trading Agent
==========================================
A reinforcement learning agent that learns to make buy/sell/hold decisions
in a simulated stock trading environment using Deep Q-Networks (DQN).

Author: Student
Course: Reinforcement Learning Assignment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import warnings

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CUSTOM TRADING ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class StockTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for stock trading simulation.

    The agent observes market indicators and makes trading decisions:
    - Action 0: Hold (do nothing)
    - Action 1: Buy (purchase shares)
    - Action 2: Sell (sell shares)

    The reward is based on portfolio value changes and trading performance.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_fee=0.001):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        # State features: [balance_norm, shares_norm, price_norm, rsi, macd, bb_position, volume_norm]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.portfolio_values = [self.initial_balance]
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self):
        """Get current state observation."""
        row = self.df.iloc[self.current_step]

        # Normalize features
        balance_norm = self.balance / self.initial_balance
        shares_norm = self.shares_held / 100  # Normalize by typical holding
        price_norm = row['Close'] / self.df['Close'].mean()
        rsi_norm = row['RSI'] / 100 if 'RSI' in row else 0.5
        macd_norm = row['MACD_Hist'] / self.df['MACD_Hist'].std() if 'MACD_Hist' in row else 0
        bb_position = row['BB_Position'] if 'BB_Position' in row else 0.5
        volume_norm = row['Volume_Norm'] if 'Volume_Norm' in row else 1.0

        return np.array([
            balance_norm, shares_norm, price_norm,
            rsi_norm, macd_norm, bb_position, volume_norm
        ], dtype=np.float32)

    def step(self, action):
        """Execute one step in the environment."""
        current_price = self.df.iloc[self.current_step]['Close']
        prev_portfolio_value = self.balance + self.shares_held * current_price

        # Execute action
        if action == 1:  # Buy
            shares_to_buy = int(self.balance * 0.5 / (current_price * (1 + self.transaction_fee)))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_shares_bought += shares_to_buy
                self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_to_sell = self.shares_held
                revenue = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.balance += revenue
                self.shares_held = 0
                self.total_shares_sold += shares_to_sell
                self.trades.append(('SELL', self.current_step, current_price, shares_to_sell))

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= len(self.df) - 1

        # Calculate new portfolio value
        if not done:
            new_price = self.df.iloc[self.current_step]['Close']
        else:
            new_price = current_price

        new_portfolio_value = self.balance + self.shares_held * new_price
        self.portfolio_values.append(new_portfolio_value)

        # Calculate reward (percentage change in portfolio value)
        reward = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value * 100

        # Bonus for profitable trades, penalty for excessive trading
        if len(self.trades) > 0 and self.trades[-1][0] == 'SELL':
            if new_portfolio_value > self.initial_balance:
                reward += 0.5  # Bonus for profitable position

        obs = self._get_observation() if not done else np.zeros(7, dtype=np.float32)

        return obs, reward, done, False, {'portfolio_value': new_portfolio_value}

    def render(self):
        """Render current state."""
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.shares_held * current_price
        print(f"Step: {self.current_step} | Balance: ${self.balance:.2f} | "
              f"Shares: {self.shares_held} | Portfolio: ${portfolio_value:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DEEP Q-NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class DQN(nn.Module):
    """
    Deep Q-Network with dueling architecture.

    The network outputs Q-values for each possible action given a state.
    Uses fully connected layers with ReLU activation.
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine value and advantage (dueling DQN)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores (state, action, reward, next_state, done) tuples and allows
    random sampling for training the DQN.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DQN AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Deep Q-Learning Agent with experience replay and target network.

    Uses epsilon-greedy exploration strategy and trains using temporal
    difference learning with a target network for stability.
    """

    def __init__(self, state_size, action_size, learning_rate=0.001,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64,
                 target_update=10):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)

        # Training metrics
        self.losses = []
        self.episode_rewards = []
        self.epsilon_history = []

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = self.criterion(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DATA GENERATION AND PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_stock_data(n_days=1000, seed=42):
    """
    Generate synthetic stock price data with realistic patterns.

    Creates OHLCV data with trend, volatility clustering, and mean reversion.
    """
    np.random.seed(seed)

    # Generate price using geometric Brownian motion with mean reversion
    dt = 1 / 252  # Daily timestep
    mu = 0.08  # Annual drift
    sigma = 0.20  # Annual volatility

    prices = [100.0]  # Starting price

    for _ in range(n_days - 1):
        # Add regime changes
        if random.random() < 0.01:
            sigma = np.clip(sigma + random.uniform(-0.05, 0.05), 0.10, 0.40)

        # GBM with mean reversion
        mean_price = 100
        reversion_speed = 0.02
        drift = mu * dt + reversion_speed * (mean_price - prices[-1]) / prices[-1] * dt
        shock = sigma * np.sqrt(dt) * np.random.randn()

        new_price = prices[-1] * np.exp(drift + shock)
        prices.append(new_price)

    prices = np.array(prices)

    # Generate OHLCV data
    data = {
        'Date': pd.date_range(start='2020-01-01', periods=n_days, freq='B'),
        'Open': prices * (1 + np.random.randn(n_days) * 0.005),
        'High': prices * (1 + np.abs(np.random.randn(n_days) * 0.015)),
        'Low': prices * (1 - np.abs(np.random.randn(n_days) * 0.015)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n_days)
    }

    df = pd.DataFrame(data)
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)

    return df


def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + 2 * std_20
    df['BB_Lower'] = sma_20 - 2 * std_20
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)

    # Volume normalization
    df['Volume_Norm'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # Drop NaN rows
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_agent(env, agent, n_episodes=200, verbose=True):
    """
    Train the DQN agent on the trading environment.

    Returns training history including rewards and portfolio values.
    """
    episode_rewards = []
    episode_portfolio_values = []
    best_portfolio_value = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, info = env.step(action)

            # Store transition and train
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            total_reward += reward

        # Update target network periodically
        if episode % agent.target_update == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        final_portfolio_value = env.portfolio_values[-1]
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(final_portfolio_value)

        # Track best performance
        if final_portfolio_value > best_portfolio_value:
            best_portfolio_value = final_portfolio_value

        if verbose and (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            avg_portfolio = np.mean(episode_portfolio_values[-20:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Portfolio: ${avg_portfolio:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return {
        'rewards': episode_rewards,
        'portfolio_values': episode_portfolio_values,
        'losses': agent.losses,
        'epsilon_history': agent.epsilon_history
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: EVALUATION AND VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_agent(env, agent, n_episodes=10):
    """Evaluate trained agent performance."""
    results = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            next_state, _, done, _, info = env.step(action)
            state = next_state

        final_value = env.portfolio_values[-1]
        total_return = (final_value - env.initial_balance) / env.initial_balance * 100
        results.append({
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': len(env.trades),
            'portfolio_history': env.portfolio_values.copy()
        })

    return results


def plot_training_results(history, save_path='training_results.png'):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(history['rewards'], alpha=0.6, color='blue')
    window = min(20, len(history['rewards']))
    if len(history['rewards']) >= window:
        smoothed = pd.Series(history['rewards']).rolling(window).mean()
        ax1.plot(smoothed, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax1.set_title('Episode Rewards Over Training', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Portfolio Values
    ax2 = axes[0, 1]
    ax2.plot(history['portfolio_values'], alpha=0.6, color='green')
    ax2.axhline(y=10000, color='red', linestyle='--', label='Initial Balance')
    if len(history['portfolio_values']) >= window:
        smoothed = pd.Series(history['portfolio_values']).rolling(window).mean()
        ax2.plot(smoothed, color='darkgreen', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax2.set_title('Portfolio Value Over Training', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Training Loss
    ax3 = axes[1, 0]
    if len(history['losses']) > 0:
        ax3.plot(history['losses'], alpha=0.3, color='orange')
        loss_window = min(100, len(history['losses']))
        if len(history['losses']) >= loss_window:
            smoothed_loss = pd.Series(history['losses']).rolling(loss_window).mean()
            ax3.plot(smoothed_loss, color='red', linewidth=2, label=f'{loss_window}-Step Moving Avg')
    ax3.set_title('Training Loss (Huber Loss)', fontsize=12)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Epsilon Decay
    ax4 = axes[1, 1]
    ax4.plot(history['epsilon_history'], color='purple')
    ax4.set_title('Exploration Rate (Epsilon) Decay', fontsize=12)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Training results saved to {save_path}")


def plot_evaluation_results(env, eval_results, save_path='evaluation_results.png'):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Portfolio trajectory from best episode
    best_idx = np.argmax([r['total_return'] for r in eval_results])
    best_result = eval_results[best_idx]

    ax1 = axes[0, 0]
    ax1.plot(best_result['portfolio_history'], color='green', linewidth=2)
    ax1.axhline(y=10000, color='red', linestyle='--', label='Initial Balance', alpha=0.7)
    ax1.fill_between(range(len(best_result['portfolio_history'])),
                     10000, best_result['portfolio_history'],
                     alpha=0.3, color='green' if best_result['total_return'] > 0 else 'red')
    ax1.set_title(f'Best Episode Portfolio Trajectory (Return: {best_result["total_return"]:.2f}%)', fontsize=12)
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Stock price with buy/sell markers
    ax2 = axes[0, 1]
    ax2.plot(env.df['Close'].values, color='black', linewidth=1, label='Stock Price')
    ax2.set_title('Stock Price Movement', fontsize=12)
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Returns distribution
    ax3 = axes[1, 0]
    returns = [r['total_return'] for r in eval_results]
    ax3.hist(returns, bins=10, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_title('Distribution of Returns Across Episodes', fontsize=12)
    ax3.set_xlabel('Total Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_text = (
        f"Evaluation Summary ({len(eval_results)} Episodes)\n"
        f"{'=' * 40}\n\n"
        f"Average Return:     {np.mean(returns):>8.2f}%\n"
        f"Best Return:        {np.max(returns):>8.2f}%\n"
        f"Worst Return:       {np.min(returns):>8.2f}%\n"
        f"Std Dev:            {np.std(returns):>8.2f}%\n"
        f"Win Rate:           {sum(1 for r in returns if r > 0) / len(returns) * 100:>8.1f}%\n\n"
        f"Average Final Value: ${np.mean([r['final_value'] for r in eval_results]):>10.2f}\n"
        f"Average Trades:      {np.mean([r['num_trades'] for r in eval_results]):>10.1f}"
    )

    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Performance Statistics', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Evaluation results saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main function to run the trading agent."""
    print("=" * 70)
    print("TradeMaster - Deep Q-Network Trading Agent")
    print("=" * 70)

    # Generate and preprocess data
    print("\n[1/5] Generating synthetic stock data...")
    df = generate_synthetic_stock_data(n_days=1000, seed=42)
    df = add_technical_indicators(df)
    print(f"      Dataset: {len(df)} trading days with {df.shape[1]} features")

    # Split data into train and test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    print(f"      Training: {len(train_df)} days | Testing: {len(test_df)} days")

    # Create environments
    print("\n[2/5] Creating trading environments...")
    train_env = StockTradingEnv(train_df, initial_balance=10000)
    test_env = StockTradingEnv(test_df, initial_balance=10000)

    # Initialize agent
    print("\n[3/5] Initializing DQN agent...")
    agent = DQNAgent(
        state_size=7,
        action_size=3,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10
    )
    print(f"      Device: {agent.device}")
    print(f"      Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")

    # Train agent
    print("\n[4/5] Training agent...")
    print("-" * 70)
    history = train_agent(train_env, agent, n_episodes=200, verbose=True)
    print("-" * 70)

    # Evaluate agent
    print("\n[5/5] Evaluating agent on test data...")
    eval_results = evaluate_agent(test_env, agent, n_episodes=20)

    # Print final results
    returns = [r['total_return'] for r in eval_results]
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Average Return:  {np.mean(returns):>8.2f}%")
    print(f"Best Return:     {np.max(returns):>8.2f}%")
    print(f"Worst Return:    {np.min(returns):>8.2f}%")
    print(f"Win Rate:        {sum(1 for r in returns if r > 0) / len(returns) * 100:>8.1f}%")
    print("=" * 70)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_results(history, save_path='training_results.png')
    plot_evaluation_results(test_env, eval_results, save_path='evaluation_results.png')

    # Save model
    agent.save('trademaster_model.pth')
    print("\nModel saved to trademaster_model.pth")

    return agent, history, eval_results


if __name__ == "__main__":
    agent, history, eval_results = main()