import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler


def run():
    # ── Section 1: Data Collection ──────────────────────────
    df = yf.download("GC=F", start="2015-01-01", end="2025-01-01", auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # ── Section 2: Feature Engineering ─────────────────────
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    for lag in [1, 3, 5]:
        df[f'Return_Lag_{lag}'] = df['Log_Return'].shift(lag)

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9, adjust=False).mean()

    # Bollinger Band Width and Volatility
    df['BB_Width'] = (4 * df['Close'].rolling(window=20).std()) / df['Close'].rolling(window=20).mean()
    df['Vol_10'] = df['Log_Return'].rolling(window=10).std()

    # Target
    df['Target'] = df['Log_Return'].shift(-1)

    # Store full history before dropping rows
    full_history = df['Close'].copy()

    df.dropna(inplace=True)

    # ── Drop raw OHLCV to prevent leakage ───────────────────
    df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

    close_prices = df['Close']

    # ── Section 3: Train/Test Split & Scaling ───────────────
    features = [col for col in df.columns if col not in ['Target', 'Close']]

    X, y = df[features], df['Target']

    split_idx = int(len(df) * 0.80)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    prices_test = close_prices.iloc[split_idx:]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # ── Section 4: Model Training ───────────────────────────
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_ret_preds = rf.predict(X_test_s)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_ret_preds = lr.predict(X_test_s)

    # ── Section 5: Price Reconstruction & Metrics ──────────
    rf_price_preds = prices_test.values * np.exp(rf_ret_preds)
    lr_price_preds = prices_test.values * np.exp(lr_ret_preds)
    actual_prices = prices_test.values * np.exp(y_test.values)

    # Calculate metrics and store them
    metrics = {}
    for name, pred in [("Random Forest", rf_price_preds), ("Linear Regression", lr_price_preds)]:
        mae = mean_absolute_error(actual_prices, pred)
        rmse = np.sqrt(mean_squared_error(actual_prices, pred))
        r2 = r2_score(actual_prices, pred)
        metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print(f"\n{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R² Score':<10}")
    print("-" * 60)

    for name, m in metrics.items():
        print(f"{name:<20} | ${m['MAE']:>8.2f} | ${m['RMSE']:>8.2f} | {m['R2']:>9.4f}")

    # ── Section 6: Visualization ────────────────────────────
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: 10-Year History
    ax1.plot(full_history.index, full_history.values, color='black', label='Gold Price (XAU/USD)')
    ax1.axvline(prices_test.index[0], color='red', linestyle='--', label='Validation Start')
    ax1.set_title('10-Year Gold Price Trend (2015 - 2025)', fontsize=14)
    ax1.set_ylabel('Price (USD)')
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax1.legend()

    # Plot 2: 120-Day Zoom with Metrics
    zoom = -120
    ax2.plot(actual_prices[zoom:], label='Actual Price', color='black', linewidth=2)
    ax2.plot(rf_price_preds[zoom:], label='RF Predicted', color='gold', linestyle='--')
    ax2.plot(lr_price_preds[zoom:], label='LR Predicted', color='steelblue', linestyle=':')
    ax2.set_title('Model Performance Zoom (Recent 120 Days)', fontsize=14)
    ax2.set_ylabel('Price (USD)')
    ax2.set_xlabel('Trading Days')
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax2.legend(loc='upper left')

    # Add metrics text box - FIXED formatting
    rf_m = metrics['Random Forest']
    lr_m = metrics['Linear Regression']

    metrics_text = (
        f"Random Forest\n"
        f"  MAE:   ${rf_m['MAE']:.2f}\n"
        f"  RMSE: ${rf_m['RMSE']:.2f}\n"
        f"  R²:      {rf_m['R2']:.4f}\n"
        f"\n"
        f"Linear Regression\n"
        f"  MAE:   ${lr_m['MAE']:.2f}\n"
        f"  RMSE: ${lr_m['RMSE']:.2f}\n"
        f"  R²:      {lr_m['R2']:.4f}"
    )

    # Position the text box in the lower left of the bottom plot (away from data)
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    ax2.text(0.02, 0.02, metrics_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=props, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=150)
    plt.show()

    # ── Section 7: Feature Importance ───────────────────────
    importances = rf.feature_importances_
    feat_names = X.columns.tolist()

    feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=feat_df, x='Importance', y='Feature', hue='Feature', legend=False, palette='YlOrBr')
    plt.title('Top 10 Feature Importances — Random Forest')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    run()