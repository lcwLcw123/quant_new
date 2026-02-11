# 华鼎股份量化交易系统 - 完整版
# 深度学习LSTM模型 + 15分钟K线数据

import akshare as ak
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
CONFIG = {
    'stock_code': '601113',  # 华鼎股份
    'stock_name': '华鼎股份',
    'data_period': '15',     # 15分钟K线
    'start_date': '20240101',
    'end_date': '20250101',
    'sequence_length': 16,   # 使用16个15分钟数据(4小时)预测
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'prediction_target': 'next_15min_close',  # 预测下个15分钟收盘价
    'threshold': 0.005,  # 买入信号阈值(0.5%)
}

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print(f"华鼎股份({CONFIG['stock_code']})量化交易系统")
print(f"深度学习模型 + {CONFIG['data_period']}分钟K线数据")
print("="*70)

# ==================== 1. 数据获取 ====================
print("\n[STEP 1/6] 获取数据...")

def generate_dummy_data():
    """生成模拟数据用于演示"""
    np.random.seed(42)
    n = 3000
    dates = pd.date_range(start='2024-01-01', periods=n, freq='15min')
    
    # 模拟价格走势
    prices = []
    base_price = 8.5
    for i in range(n):
        random_walk = np.random.normal(0, 0.02)
        base_price = base_price * (1 + random_walk)
        prices.append(base_price)
    
    df = pd.DataFrame({
        '日期': dates,
        '开盘': prices,
        '收盘': prices,
        '最高': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        '最低': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        '成交量': np.random.randint(100000, 10000000, n),
    })
    return df

try:
    stock_df = ak.stock_zh_a_hist(symbol=CONFIG['stock_code'], period=CONFIG['data_period'], 
                                  start_date=CONFIG['start_date'], end_date=CONFIG['end_date'])
    stock_df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
    print(f"✓ 成功获取 {len(stock_df)} 条15分钟K线数据")
    print(f"  时间范围: {stock_df['日期'].iloc[0]} ~ {stock_df['日期'].iloc[-1]}")
except Exception as e:
    print(f"⚠ 数据获取失败: {e}")
    print("使用模拟数据进行演示...")
    stock_df = generate_dummy_data()
    print(f"✓ 生成了 {len(stock_df)} 条模拟15分钟K线数据")

# ==================== 2. 特征工程 ====================
print("\n[STEP 2/6] 特征工程...")

def create_features(df):
    """创建技术指标特征"""
    df = df.copy()
    
    # 价格变化
    df['price_change'] = df['收盘'].pct_change()
    df['price_change_1'] = df['收盘'].pct_change(periods=1)
    df['price_change_4'] = df['收盘'].pct_change(periods=4)   # 1小时
    df['price_change_8'] = df['收盘'].pct_change(periods=8)   # 2小时
    
    # 移动平均线
    df['ma_5'] = df['收盘'].rolling(window=5).mean()
    df['ma_10'] = df['收盘'].rolling(window=10).mean()
    df['ma_20'] = df['收盘'].rolling(window=20).mean()
    
    # 相对价格位置
    df['price_vs_ma5'] = (df['收盘'] - df['ma_5']) / df['ma_5']
    df['price_vs_ma10'] = (df['收盘'] - df['ma_10']) / df['ma_10']
    
    # 波动率
    df['volatility_5'] = df['price_change'].rolling(window=5).std()
    df['volatility_10'] = df['price_change'].rolling(window=10).std()
    
    # 成交量特征
    df['volume_change'] = df['成交量'].pct_change()
    df['volume_ma5'] = df['成交量'].rolling(window=5).mean()
    df['volume_ratio'] = df['成交量'] / df['volume_ma5']
    
    # RSI
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['收盘'].ewm(span=12, adjust=False).mean()
    exp2 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 布林带
    df['bb_middle'] = df['收盘'].rolling(window=20).mean()
    df['bb_std'] = df['收盘'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['收盘'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 价格动量
    df['momentum'] = df['收盘'] - df['收盘'].shift(8)
    
    # 高低价范围
    df['hl_range'] = (df['最高'] - df['最低']) / df['收盘']
    
    return df

stock_df = create_features(stock_df)

# 创建目标变量：下个15分钟收盘价
stock_df['next_close'] = stock_df['收盘'].shift(-1)
stock_df['target_return'] = (stock_df['next_close'] - stock_df['收盘']) / stock_df['收盘']

# 移除NaN
feature_cols = ['收盘', 'price_change', 'price_change_1', 'price_change_4',
                'ma_5', 'ma_10', 'price_vs_ma5', 'volatility_5',
                'volume_change', 'volume_ratio', 'rsi', 'macd', 'macd_hist',
                'bb_position', 'momentum', 'hl_range']
df_clean = stock_df.dropna(subset=feature_cols + ['target_return'])
print(f"✓ 特征工程完成，有效样本: {len(df_clean)}")

# ==================== 3. 数据准备 ====================
print("\n[STEP 3/6] 数据准备...")

# 准备特征和目标
X = df_clean[feature_cols].values
y = df_clean['target_return'].values

# 标准化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# 创建序列数据
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, CONFIG['sequence_length'])

# 划分训练/验证/测试集
train_size = int(len(X_seq) * CONFIG['train_ratio'])
val_size = int(len(X_seq) * CONFIG['val_ratio'])

X_train, y_train = X_seq[:train_size], y_seq[:train_size]
X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

# 转换为PyTorch张量
X_train_t = torch.FloatTensor(X_train).transpose(1, 2)  # (batch, features, seq_len)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val).transpose(1, 2)
y_val_t = torch.FloatTensor(y_val)
X_test_t = torch.FloatTensor(X_test).transpose(1, 2)
y_test_t = torch.FloatTensor(y_test)

# 创建DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

print(f"✓ 训练集: {len(X_train)} 样本")
print(f"✓ 验证集: {len(X_val)} 样本")
print(f"✓ 测试集: {len(X_test)} 样本")

# ==================== 4. LSTM模型 ====================
print("\n[STEP 4/6] 构建LSTM深度学习模型...")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

model = LSTMModel(
    input_size=len(feature_cols),
    hidden_size=CONFIG['hidden_size'],
    num_layers=CONFIG['num_layers'],
    dropout=CONFIG['dropout']
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

print(model)
print(f"✓ 模型参数总量: {sum(p.numel() for p in model.parameters()):,}")

# ==================== 5. 模型训练 ====================
print("\n[STEP 5/6] 训练模型...")

best_val_loss = float('inf')
patience_counter = 0
early_stop_patience = 20
train_losses, val_losses = [], []

for epoch in range(CONFIG['epochs']):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    
    # 验证
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t).item()
    
    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        best_epoch = epoch + 1
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1}/{CONFIG['epochs']}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if patience_counter >= early_stop_patience:
        print(f"✓ 早停于第 {epoch+1} 个epoch")
        break

print(f"✓ 最佳模型保存于 Epoch {best_epoch}")

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# ==================== 6. 回测 ====================
print("\n[STEP 6/6] 执行回测...")

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_t).numpy()

# 反标准化
y_pred_original = scaler_y.inverse_transform(y_pred_test)
y_test_original = scaler_y.inverse_transform(y_test)

# 计算预测指标
mse = mean_squared_error(y_test_original, y_pred_original)
mae = mean_absolute_error(y_test_original, y_pred_original)
print(f"✓ MSE: {mse:.8f}")
print(f"✓ MAE: {mae:.8f}")

# 生成交易信号
df_backtest = df_clean.iloc[train_size+val_size+CONFIG['sequence_length']:].copy().reset_index(drop=True)
df_backtest['predicted_return'] = y_pred_original.flatten()
df_backtest['signal'] = 0
df_backtest.loc[df_backtest['predicted_return'] > CONFIG['threshold'], 'signal'] = 1   # 买入
df_backtest.loc[df_backtest['predicted_return'] < -CONFIG['threshold'], 'signal'] = -1  # 卖出

# 回测逻辑
initial_capital = 100000  # 初始资金10万
capital = initial_capital
position = 0  # 持仓股数
shares = 0
trade_count = 0
wins = 0
losses = 0
total_profit = 0
total_loss = 0
trades = []

for i in range(len(df_backtest)):
    signal = df_backtest.iloc[i]['signal']
    close_price = df_backtest.iloc[i]['收盘']
    
    if signal == 1 and position == 0:  # 买入信号且无持仓
        shares = capital // (close_price * 100) * 100  # 按100股整数倍买入
        if shares > 0:
            position = 1
            buy_price = close_price
            capital -= shares * buy_price
            trade_count += 1
            trades.append({'type': 'BUY', 'price': buy_price, 'time': df_backtest.iloc[i]['日期']})
    
    elif signal == -1 and position == 1:  # 卖出信号且有持仓
        sell_price = close_price
        profit = (sell_price - buy_price) * shares
        capital += shares * sell_price
        
        if profit > 0:
            wins += 1
            total_profit += profit
        else:
            losses += 1
            total_loss += abs(profit)
        
        trade_count += 1
        trades.append({'type': 'SELL', 'price': sell_price, 'time': df_backtest.iloc[i]['日期'],
                      'profit': profit, 'return': (sell_price - buy_price) / buy_price * 100})
        position = 0
        shares = 0

# 如果还有持仓，按最后价格计算
if position == 1:
    final_price = df_backtest.iloc[-1]['收盘']
    unrealized_profit = (final_price - buy_price) * shares
    capital += unrealized_profit
    trades.append({'type': 'CLOSE', 'price': final_price, 'time': df_backtest.iloc[-1]['日期'],
                  'profit': unrealized_profit, 'return': (final_price - buy_price) / buy_price * 100})

# 计算回测指标
total_return = (capital - initial_capital) / initial_capital * 100
annualized_return = total_return * (252*16) / len(df_backtest)  # 每天16个15分钟
win_rate = wins / trade_count * 2 if trade_count > 0 else 0  # 买卖各一次算一笔
profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
max_capital = initial_capital
max_drawdown = 0
current_capital = initial_capital
drawdowns = []

for trade in trades:
    if trade['type'] in ['SELL', 'CLOSE']:
        current_capital += trade['profit']
        max_capital = max(max_capital, current_capital)
        drawdown = (max_capital - current_capital) / max_capital * 100
        max_drawdown = max(max_drawdown, drawdown)
        drawdowns.append(drawdown)

# 计算夏普比率
returns = [t['return'] for t in trades if 'return' in t]
sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252*16) if len(returns) > 1 and np.std(returns) > 0 else 0

print("\n" + "="*70)
print("                   华鼎股份量化交易系统 - 回测报告")
print("="*70)

print(f"""
【基本信息】
  股票代码: {CONFIG['stock_code']} ({CONFIG['stock_name']})
  数据周期: 15分钟K线
  回测期间: {df_backtest['日期'].iloc[0]} ~ {df_backtest['日期'].iloc[-1]}
  样本数量: {len(df_backtest)} 个15分钟周期

【模型配置】
  模型类型: LSTM (长短期记忆网络)
  序列长度: {CONFIG['sequence_length']} 个时间步 (4小时)
  隐藏层大小: {CONFIG['hidden_size']}
  LSTM层数: {CONFIG['num_layers']}
  Dropout: {CONFIG['dropout']}
  特征数量: {len(feature_cols)} 个

【回测结果】
  初始资金: ¥{initial_capital:,.2f}
  最终资金: ¥{capital:,.2f}
  总收益率: {total_return:.2f}%
  年化收益率: {annualized_return:.2f}%
  交易次数: {trade_count // 2} 笔完整交易
  胜率: {win_rate:.2f}%
  盈亏比: {profit_factor:.2f}
  最大回撤: {max_drawdown:.2f}%
  夏普比率: {sharpe_ratio:.2f}

【预测性能】
  MSE: {mse:.8f}
  MAE: {mae:.8f}
""")

# 生成图表
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. 预测 vs 实际
ax1 = axes[0]
ax1.plot(y_test_original[:200], label='实际收益', alpha=0.8)
ax1.plot(y_pred_original[:200], label='预测收益', alpha=0.8)
ax1.set_title('LSTM预测 vs 实际收益 (前200个样本)')
ax1.set_xlabel('样本')
ax1.set_ylabel('收益率')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 损失曲线
ax2 = axes[1]
ax2.plot(train_losses, label='训练损失')
ax2.plot(val_losses, label='验证损失')
ax2.set_title('训练过程损失曲线')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 资产曲线
ax3 = axes[2]
capital_curve = [initial_capital]
for trade in trades:
    if 'profit' in trade:
        capital_curve.append(capital_curve[-1] + trade['profit'])
ax3.plot(capital_curve)
ax3.set_title(f'资产曲线 (总收益: {total_return:.2f}%)')
ax3.set_xlabel('交易次数')
ax3.set_ylabel('资金 (¥)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_report.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存: backtest_report.png")

# 保存回测数据
df_backtest.to_csv('backtest_data.csv', index=False)
print("✓ 回测数据已保存: backtest_data.csv")

# 保存交易记录
trades_df = pd.DataFrame(trades)
trades_df.to_csv('trades_history.csv', index=False)
print("✓ 交易记录已保存: trades_history.csv")

print("\n" + "="*70)
print("                           回测完成!")
print("="*70)
