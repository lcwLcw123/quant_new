#!/usr/bin/env python3
# 华鼎股份量化交易系统 - 轻量版 (不需要torch)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*75)
print("           华鼎股份(601113) 量化交易系统 - 轻量版")
print("="*75)

# 配置
CONFIG = {
    'stock_code': '601113',
    'stock_name': '华鼎股份',
    'initial_capital': 100000,
    'buy_threshold': 0.005,
    'sell_threshold': -0.005,
}

# 1. 模拟/获取数据
print("\n[1/4] 数据准备...")
print("使用模拟数据进行演示...")

n_samples = 2000
dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
prices = []
base = 8.5
for i in range(n_samples):
    base *= (1 + np.random.normal(0.0003, 0.015))
    prices.append(base)

df = pd.DataFrame({'日期': dates, '收盘': prices})

# 2. 特征工程
print("[2/4] 特征工程...")
def create_features(df):
    df = df.copy()
    df['return_1'] = df['收盘'].pct_change(1)
    df['return_4'] = df['收盘'].pct_change(4)  # 1小时
    df['return_8'] = df['收盘'].pct_change(8)  # 2小时
    df['ma_5'] = df['收盘'].rolling(5).mean()
    df['ma_10'] = df['收盘'].rolling(10).mean()
    df['volatility'] = df['return_1'].rolling(5).std()
    df['volume_ratio'] = df['成交量'] = np.random.randint(1000000, 10000000, len(df))
    df['volume_ratio'] = df['成交量'].rolling(5).mean()
    df['rsi'] = 50 + np.random.randn(len(df)) * 20  # 简化RSI
    df['macd'] = df['收盘'].ewm(12).mean() - df['收盘'].ewm(26).mean()
    df['bb_pos'] = (df['收盘'] - df['收盘'].rolling(20).mean()) / (df['收盘'].rolling(20).std() + 1e-8)
    df['momentum'] = df['收盘'] - df['收盘'].shift(8)
    return df

df = create_features(df)
df['next_return'] = df['收盘'].pct_change().shift(-1)
df_clean = df.dropna()
print(f"✓ 有效样本: {len(df_clean)}")

# 3. 模型训练
print("[3/4] 训练模型...")

features = ['return_1','return_4','return_8','ma_5','ma_10','volatility','rsi','macd','bb_pos','momentum']
X = df_clean[features].values
y = df_clean['next_return'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

train_size = int(len(X_scaled) * 0.7)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 使用梯度提升
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"✓ 模型训练完成")
print(f"  MSE:  {mse:.8f}")
print(f"  MAE:  {mae:.6f}")

# 4. 回测
print("[4/4] 执行回测...")

df_backtest = df_clean.iloc[train_size:].reset_index(drop=True)
df_backtest['predicted'] = y_pred

capital = CONFIG['initial_capital']
shares = 0
position = 0
buy_price = 0
trades = []

for i in range(len(df_backtest)):
    pred = df_backtest.iloc[i]['predicted']
    price = df_backtest.iloc[i]['收盘']
    signal = 1 if pred > CONFIG['buy_threshold'] else (-1 if pred < CONFIG['sell_threshold'] else 0)
    
    if signal == 1 and position == 0:
        shares = capital // (price * 100) * 100
        if shares > 0:
            position = 1
            buy_price = price
            capital -= shares * price
            trades.append({'type': 'BUY', 'price': price, 'time': i})
    
    elif signal == -1 and position == 1:
        profit = (price - buy_price) * shares
        capital += shares * price
        trades.append({'type': 'SELL', 'price': price, 'profit': profit})
        position = 0
        shares = 0

if position == 1:
    final = df_backtest.iloc[-1]['收盘']
    profit = (final - buy_price) * shares
    capital += profit
    trades.append({'type': 'CLOSE', 'price': final, 'profit': profit})

# 计算指标
total_return = (capital - CONFIG['initial_capital']) / CONFIG['initial_capital'] * 100
wins = sum(1 for t in trades if t.get('profit', 0) > 0)
losses = sum(1 for t in trades if t.get('profit', 0) < 0)
win_rate = wins / ((wins + losses) or 1) * 100

# 输出报告
print("\n" + "="*75)
print("                   华鼎股份量化交易系统 - 回测报告")
print("="*75)
print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                           基本信息                                       │
├─────────────────────────────────────────────────────────────────────────┤
│   股票代码:     {CONFIG['stock_code']} ({CONFIG['stock_name']})                            │
│   数据周期:     15分钟K线                                                 │
│   回测样本:     {len(df_backtest)} 个周期                                        │
│   模型类型:     Gradient Boosting                                        │
│   特征数量:     {len(features)} 个                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                           资金状况                                       │
├─────────────────────────────────────────────────────────────────────────┤
│   初始资金:     ¥{CONFIG['initial_capital']:>12,.2f}                                      │
│   最终资金:     ¥{capital:>12,.2f}                                      │
│   总收益率:     {total_return:>+12.2f}%                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                           交易统计                                       │
├─────────────────────────────────────────────────────────────────────────┤
│   交易次数:     {len(trades):>12} 笔                                           │
│   盈利次数:     {wins:>12} 次                                              │
│   亏损次数:     {losses:>12} 次                                              │
│   胜率:         {win_rate:>12.2f}%                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                           模型性能                                       │
├─────────────────────────────────────────────────────────────────────────┤
│   MSE:          {mse:>12.8f}                                         │
│   MAE:          {mae:>12.6f}                                           │
│   预测方向准确: {(np.sign(y_pred) == np.sign(y_test)).mean()*100:>12.2f}%                                          │
└─────────────────────────────────────────────────────────────────────────┘
""")

# 交易记录
print("【最近10笔交易】")
print("  #   类型      价格        收益")
print("  ─────────────────────────────────────")
for i, t in enumerate(trades[-10:]):
    profit_str = f"+¥{t['profit']:.2f}" if t.get('profit',0) > 0 else (f"¥{t['profit']:.2f}" if t.get('profit',0) < 0 else "-")
    print(f"  {i+1:2d}  {t['type']:<6}  ¥{t['price']:>8.2f}  {profit_str:>12}")

print("\n" + "="*75)
print("                         回测完成!")
print("="*75)

# 保存数据
df_backtest.to_csv('backtest_data.csv', index=False)
pd.DataFrame(trades).to_csv('trades.csv', index=False)
print("\n✓ 数据已保存: backtest_data.csv, trades.csv")
