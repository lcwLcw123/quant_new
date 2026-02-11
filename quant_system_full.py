#!/usr/bin/env python3
# 华鼎股份量化交易系统 - 完整版 (支持在线/离线模式)
# 运行: python3 quant_system_full.py

import sys
import os

# 检查依赖
def check_dependencies():
    missing = []
    for mod in ['torch', 'akshare', 'sklearn', 'pandas', 'numpy', 'matplotlib']:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing

missing = check_dependencies()
if missing:
    print(f"缺少依赖: {', '.join(missing)}")
    print("请安装: pip install torch sklearn akshare pandas numpy matplotlib ta")
    print("\n使用轻量模式运行...")
    exec(open('quant_lightweight.py').read())
    sys.exit(0)

# ==================== 完整版开始 ====================
import akshare as ak
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

# 配置
CONFIG = {
    'stock_code': '601113',
    'stock_name': '华鼎股份',
    'data_period': '15',
    'start_date': '20240101',
    'end_date': '20250101',
    'sequence_length': 16,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'initial_capital': 100000,
    'buy_threshold': 0.005,
    'sell_threshold': -0.005,
}

print("="*70)
print(f"华鼎股份({CONFIG['stock_code']}) 量化交易系统 - LSTM深度学习版")
print("="*70)

# 1. 获取数据
print("\n[1/5] 获取数据...")
try:
    stock_df = ak.stock_zh_a_hist(symbol=CONFIG['stock_code'], 
                                  period=CONFIG['data_period'],
                                  start_date=CONFIG['start_date'], 
                                  end_date=CONFIG['end_date'])
    stock_df.columns = ['日期','开盘','收盘','最高','最低','成交量','成交额','振幅','涨跌幅','涨跌额','换手率']
    print(f"✓ 成功获取 {len(stock_df)} 条数据")
except Exception as e:
    print(f"⚠ 获取失败: {e}")
    print("使用模拟数据...")
    stock_df = pd.DataFrame({
        '日期': pd.date_range('2024-01-01', periods=2500, freq='15min'),
        '收盘': np.cumsum(np.random.randn(2500) * 0.02 + 0.0003) + 8.5,
    })

# 2. 特征工程
print("\n[2/5] 特征工程...")
def create_features(df):
    df = df.copy()
    df['price_change'] = df['收盘'].pct_change()
    df['price_change_4'] = df['收盘'].pct_change(4)
    df['ma_5'] = df['收盘'].rolling(5).mean()
    df['ma_10'] = df['收盘'].rolling(10).mean()
    df['volatility_5'] = df['price_change'].rolling(5).std()
    df['volume_ratio'] = df['成交量'].rolling(5).mean() / df['成交量'].rolling(5).mean()
    df['rsi'] = 100 - (100 / (1 + df['收盘'].pct_change().rolling(14).mean() / (df['收盘'].pct_change().rolling(14).std() + 1e-8)))
    df['macd'] = df['收盘'].ewm(12).mean() - df['收盘'].ewm(26).mean()
    df['bb_position'] = (df['收盘'] - df['收盘'].rolling(20).mean()) / (2 * df['收盘'].rolling(20).std())
    df['momentum'] = df['收盘'] - df['收盘'].shift(8)
    return df

stock_df = create_features(stock_df)
stock_df['next_return'] = stock_df['收盘'].pct_change().shift(-1)
df_clean = stock_df.dropna()
print(f"✓ 有效样本: {len(df_clean)}")

# 3. 数据准备
print("\n[3/5] 数据准备...")
features = ['收盘','price_change','price_change_4','ma_5','ma_10','volatility_5','volume_ratio','rsi','macd','bb_position','momentum']
X = df_clean[features].values
y = df_clean['next_return'].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1,1))

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, CONFIG['sequence_length'])

train_size = int(len(X_seq) * CONFIG['train_ratio'])
val_size = int(len(X_seq) * CONFIG['val_ratio'])

X_train = torch.FloatTensor(X_seq[:train_size]).transpose(1,2)
y_train = torch.FloatTensor(y_seq[:train_size])
X_val = torch.FloatTensor(X_seq[train_size:train_size+val_size]).transpose(1,2)
y_val = torch.FloatTensor(y_seq[train_size:train_size+val_size])
X_test = torch.FloatTensor(X_seq[train_size+val_size:]).transpose(1,2)
y_test = torch.FloatTensor(y_seq[train_size+val_size:])

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=CONFIG['batch_size'], shuffle=True)
print(f"✓ 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")

# 4. LSTM模型
print("\n[4/5] 构建LSTM模型...")
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return self.sigmoid(out)

model = LSTMModel(len(features), CONFIG['hidden_size'], CONFIG['num_layers'], CONFIG['dropout'])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
print(f"✓ 模型参数: {sum(p.numel() for p in model.parameters()):,}")

# 5. 训练
print("\n[5/5] 训练模型...")
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(CONFIG['epochs']):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val).item()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"✓ 早停于第 {epoch+1} 轮")
        break
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: Val Loss = {val_loss:.6f}")

model.load_state_dict(torch.load('best_model.pth'))

# 6. 回测
print("\n[回测] 执行回测...")
model.eval()
with torch.no_grad():
    y_pred = model(X_test).numpy()
y_pred = scaler_y.inverse_transform(y_pred)
y_actual = scaler_y.inverse_transform(y_test.numpy())

# 交易模拟
df_backtest = df_clean.iloc[train_size+val_size+CONFIG['sequence_length']:].reset_index(drop=True)
df_backtest['predicted'] = y_pred.flatten()

initial = CONFIG['initial_capital']
capital = initial
shares = 0
position = 0
buy_price = 0
trades = []

for i in range(len(df_backtest)):
    signal = 1 if df_backtest.iloc[i]['predicted'] > CONFIG['buy_threshold'] else (-1 if df_backtest.iloc[i]['predicted'] < CONFIG['sell_threshold'] else 0)
    price = df_backtest.iloc[i]['收盘']
    
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
        trades.append({'type': 'SELL', 'price': price, 'time': i, 'profit': profit})
        position = 0
        shares = 0

if position == 1:
    final_price = df_backtest.iloc[-1]['收盘']
    capital += shares * (final_price - buy_price)
    trades.append({'type': 'CLOSE', 'price': final_price, 'profit': shares * (final_price - buy_price)})

# 计算指标
total_return = (capital - initial) / initial * 100
wins = sum(1 for t in trades if t.get('profit', 0) > 0)
losses = sum(1 for t in trades if t.get('profit', 0) < 0)
win_rate = wins / ((wins + losses) or 1) * 100
mse = mean_squared_error(y_actual, y_pred)

# 输出报告
print("\n" + "="*70)
print("                   华鼎股份量化交易系统 - 回测报告")
print("="*70)
print(f"""
【基本信息】
  股票: {CONFIG['stock_name']}({CONFIG['stock_code']})
  周期: 15分钟K线
  样本: {len(df_backtest)} 个周期

【模型配置】
  模型: LSTM ({CONFIG['hidden_size']}隐藏层 × {CONFIG['num_layers']}层)
  序列: {CONFIG['sequence_length']} 时间步
  特征: {len(features)} 个

【回测结果】
  初始资金: ¥{initial:,.2f}
  最终资金: ¥{capital:,.2f}
  总收益:   {total_return:+.2f}%
  交易次数: {len(trades)} 笔
  胜率:     {win_rate:.2f}%
  MSE:      {mse:.8f}
""")
print("="*70)
print("✓ 回测完成!")
print("="*70)

# 保存
plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.plot(y_actual[:200], label='实际', alpha=0.8)
plt.plot(y_pred[:200], label='预测', alpha=0.8)
plt.title('LSTM预测 vs 实际收益率')
plt.legend()
plt.subplot(2,1,2)
capital_curve = [initial]
for t in trades:
    if 'profit' in t:
        capital_curve.append(capital_curve[-1] + t['profit'])
plt.plot(capital_curve)
plt.title(f'资产曲线 ({total_return:+.2f}%)')
plt.tight_layout()
plt.savefig('backtest_chart.png', dpi=150)
print("✓ 图表: backtest_chart.png")
