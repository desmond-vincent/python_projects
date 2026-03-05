# TradeMaster

A Deep Q-Network (DQN) reinforcement learning agent that learns to make autonomous buy/sell/hold decisions in a simulated stock trading environment.

## Overview

TradeMaster uses deep reinforcement learning to discover profitable trading strategies through trial-and-error interactions with market data. The agent processes technical indicators derived from historical price movements and learns to maximize portfolio returns while managing risk.

## Features

- **Dueling DQN Architecture**: Separates value and advantage streams for more stable learning
- **Double DQN**: Reduces overestimation bias in Q-value predictions
- **Experience Replay**: Breaks correlation between consecutive samples for improved training
- **Custom Trading Environment**: Gymnasium-compatible environment with realistic transaction costs
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and volatility features
- **Comprehensive Visualization**: Training curves, portfolio trajectories, and performance statistics

## Project Structure

```
TradeMaster/
├── src/
│   └── __init__.py          # Main agent implementation
├── main.py                   # Entry point
├── requirements.txt          # Dependencies
└── README.md
```

## Requirements

- Python 3.12.12
- PyTorch 2.2+
- Gymnasium 0.29+
- NumPy <2.0 (for PyTorch compatibility)
- Pandas 2.1+
- Matplotlib 3.8+
- Seaborn 0.13+

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TradeMaster
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the agent:
```bash
python main.py
```

The program will:
1. Generate synthetic stock data with realistic market dynamics
2. Train the DQN agent for 200 episodes
3. Evaluate performance on held-out test data
4. Generate visualization plots
5. Save the trained model

## Output Files

| File | Description |
|------|-------------|
| `training_results.png` | Training curves (rewards, portfolio values, loss, epsilon) |
| `evaluation_results.png` | Evaluation metrics (portfolio trajectory, returns distribution, profit statistics) |
| `trademaster_model.pth` | Saved model weights |

## Architecture

### Trading Environment

The custom Gymnasium environment simulates stock trading with:
- **State Space**: 7 normalized features (balance, shares, price, RSI, MACD, BB position, volume)
- **Action Space**: 3 discrete actions (Hold, Buy, Sell)
- **Reward**: Percentage change in portfolio value with trade bonuses

### Neural Network

```
Input (7 features)
    │
    ▼
Feature Extraction (128 → 128, ReLU)
    │
    ├──► Value Stream (64 → 1)
    │
    └──► Advantage Stream (64 → 3)
    │
    ▼
Q-Values (combined via dueling formula)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Discount Factor (γ) | 0.99 |
| Epsilon Start/End | 1.0 → 0.01 |
| Epsilon Decay | 0.995 |
| Replay Buffer Size | 10,000 |
| Batch Size | 64 |
| Target Update Frequency | 10 episodes |

## Technical Indicators

| Indicator | Description |
|-----------|-------------|
| RSI (14-day) | Momentum oscillator (0-100) |
| MACD Histogram | Trend-following momentum (12/26/9 EMA) |
| Bollinger Band Position | Price position within bands (0-1) |
| Volume Normalized | Current volume vs 20-day average |

## Sample Results

```
======================================================================
FINAL RESULTS
======================================================================
Average Return:     X.XX%
Best Return:        X.XX%
Worst Return:       X.XX%
Win Rate:           XX.X%
======================================================================
```

## Limitations

- Trained on synthetic data (may not capture real market nuances)
- Single asset trading only
- Fixed position sizing (50% of capital)
- Simplified transaction costs (no slippage/spread)

## Potential Improvements

1. Integrate real market data (Yahoo Finance, Alpha Vantage)
2. Implement continuous action spaces with PPO/A3C
3. Add multi-asset portfolio optimization
4. Incorporate sentiment analysis features
5. Use risk-adjusted rewards (Sharpe ratio)

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
- van Hasselt, H., et al. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

## License

This project is intended for academic and educational use only.