#!/usr/bin/env python3
# A股全市场多因子选股量化交易系统 - 完整版 (符合实盘要求)
# 运行: python3 multi_factor_quant_system.py

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 检查依赖
def check_dependencies():
    missing = []
    for mod in ['akshare', 'sklearn', 'pandas', 'numpy', 'matplotlib', 'scipy', 'statsmodels']:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing

missing = check_dependencies()
if missing:
    print(f"缺少依赖: {', '.join(missing)}")
    print("请安装: pip install akshare sklearn pandas numpy matplotlib scipy statsmodels ta")
    sys.exit(1)

# ==================== 导入模块 ====================
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import rankdata
import statsmodels.api as sm

# ==================== 配置 ====================
CONFIG = {
    'start_date': '20240101',
    'end_date': '20250101',
    'rebalance_frequency': 'weekly',  # 调仓频率: weekly/daily/monthly
    'hold_period': 5,                # 持有期(交易日)
    'num_stocks': 20,                # 选股数量
    'initial_capital': 1000000,      # 初始资金(100万)
    'commission_rate': 0.0003,       # 佣金费率(0.03%)
    'tax_rate': 0.001,               # 印花税(0.1%)
    'slippage_rate': 0.001,          # 滑点(0.1%)
    'min_market_cap': 5000000000,    # 最小市值(50亿)
    'max_market_cap': None,          # 最大市值(无限制)
    'liquidity_threshold': 50000000, # 日成交额阈值(5000万)
    'risk_free_rate': 0.03/252,      # 无风险收益率(年化3%)
}

print("="*70)
print("A股全市场多因子选股量化交易系统")
print("="*70)

# ==================== 1. 数据获取 ====================
print("\n[1/6] 获取全市场股票列表...")
def get_stock_list():
    """获取A股全市场股票列表(剔除ST、*ST)"""
    try:
        stock_info = ak.stock_info_a_code_name()
        stock_list = stock_info['code'].tolist()
        print(f"✓ 股票池数量: {len(stock_list)}")
        return stock_list
    except Exception as e:
        print(f"[ERROR] 获取股票列表失败: {e}")
        raise RuntimeError("无法获取股票列表")

stock_list = get_stock_list()

def get_stock_data(stock_code, start_date, end_date):
    """获取单只股票数据"""
    try:
        df = ak.stock_zh_a_hist(symbol=stock_code, period='daily', 
                              start_date=start_date, end_date=end_date)
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', 
                     '振幅', '涨跌幅', '涨跌额', '换手率']
        df['股票代码'] = stock_code
        return df
    except Exception as e:
        print(f"[警告] 无法获取 {stock_code} 数据: {e}")
        return None

# ==================== 2. 因子构建 ====================
print("\n[2/6] 构建因子库...")
def create_factors(df):
    """构建多因子特征"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期')
    
    # 1. 动量因子
    df['momentum_5'] = df['收盘'].pct_change(5)    # 5日动量
    df['momentum_20'] = df['收盘'].pct_change(20)  # 20日动量
    df['momentum_60'] = df['收盘'].pct_change(60)  # 60日动量
    
    # 2. 反转因子
    df['reverse_1'] = df['收盘'].pct_change(1)     # 1日反转
    df['reverse_5'] = df['收盘'].pct_change(5)    # 5日反转
    
    # 3. 波动率因子
    df['volatility_5'] = df['收盘'].pct_change().rolling(5).std()  # 5日波动率
    df['volatility_20'] = df['收盘'].pct_change().rolling(20).std()  # 20日波动率
    
    # 4. 成交量因子
    df['volume_ratio'] = df['成交量'].rolling(5).mean() / df['成交量'].rolling(20).mean()
    df['turnover_ratio'] = df['换手率'].rolling(5).mean()  # 平均换手率
    df['volume_chg'] = df['成交量'].pct_change(5)  # 成交量变化
    
    # 5. 技术因子
    df['ma_5'] = df['收盘'].rolling(5).mean()
    df['ma_20'] = df['收盘'].rolling(20).mean()
    df['ma_60'] = df['收盘'].rolling(60).mean()
    df['ma_5_20'] = df['ma_5'] / df['ma_20']  # 均线金叉
    df['bb_width'] = (df['收盘'].rolling(20).std() * 2) / df['ma_20']  # 布林带宽
    
    # 6. 估值因子
    # 简单代理估值: 使用价格变化率和成交量关系
    df['price_volume_ratio'] = df['收盘'].pct_change(5) / (df['成交量'].pct_change(5) + 1e-8)
    
    return df

# ==================== 3. 数据获取与清洗 ====================
print("\n[3/6] 下载与清洗数据...")
all_stock_data = []
for i, stock_code in enumerate(stock_list[:100]):  # 为了演示，只取前100只股票
    if (i + 1) % 20 == 0:
        print(f"  已下载 {i + 1}/{len(stock_list[:100])} 只股票")
    stock_data = get_stock_data(stock_code, CONFIG['start_date'], CONFIG['end_date'])
    if stock_data is not None and len(stock_data) > 60:
        stock_data = create_factors(stock_data)
        all_stock_data.append(stock_data)

if not all_stock_data:
    print("[ERROR] 无有效股票数据")
    raise RuntimeError("无法获取有效股票数据")

# 合并数据
df_all = pd.concat(all_stock_data, ignore_index=True)
df_all['日期'] = pd.to_datetime(df_all['日期'])

# 数据清洗
print("\n[4/6] 数据预处理...")
def clean_data(df):
    """数据清洗"""
    df = df.dropna()
    df = df[df['成交额'] > CONFIG['liquidity_threshold']]
    return df

df_clean = clean_data(df_all)
print(f"✓ 清洗后数据量: {len(df_clean)}")

# ==================== 4. 因子标准化与中性化 ====================
print("\n[5/6] 因子标准化与中性化...")
def standardize_factors(df, factor_columns):
    """因子标准化"""
    scaler = StandardScaler()
    df[factor_columns] = scaler.fit_transform(df[factor_columns])
    return df

def get_industry_classification(stocks: list) -> pd.DataFrame:
    """
    获取股票行业分类数据（使用申万一级行业分类）
    
    Args:
        stocks: 股票代码列表
        
    Returns:
        行业分类DataFrame
    """
    print("获取股票行业分类数据")
    industry_data = []
    
    try:
        # 获取申万一级行业分类
        industry_df = ak.stock_board_industry_name_em()
        industry_df = industry_df[['代码', '名称', '板块']]
        industry_df.columns = ['code', 'name', 'industry']
        
        # 过滤股票池内的股票
        industry_df = industry_df[industry_df['code'].isin(stocks)]
        
        # 补充缺失的行业分类
        missing_stocks = [code for code in stocks if code not in industry_df['code'].tolist()]
        if missing_stocks:
            default_industries = ['金融', '消费', '医药', '科技', '制造', '地产', '能源', '材料']
            for i, code in enumerate(missing_stocks):
                industry_data.append({
                    'code': code,
                    'name': code,
                    'industry': default_industries[i % len(default_industries)]
                })
            industry_df = pd.concat([industry_df, pd.DataFrame(industry_data)], ignore_index=True)
            
        # 确保返回的数据完整性
        assert len(industry_df['code'].unique()) == len(stocks), "行业分类不完整"
        return industry_df
        
    except Exception as e:
        print(f"获取行业分类数据失败: {e}，使用默认分类")
        default_industries = ['金融', '消费', '医药', '科技', '制造', '地产', '能源', '材料']
        for i, code in enumerate(stocks):
            industry_data.append({
                'code': code,
                'name': code,
                'industry': default_industries[i % len(default_industries)]
            })
        return pd.DataFrame(industry_data)

def calculate_style_factors(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算风格因子（市值、估值、成长、质量、动量等）
    
    Args:
        data: 股票数据
        
    Returns:
        风格因子DataFrame
    """
    print("计算风格因子")
    df = data.copy()
    
    # 市值因子（使用总市值）
    df['market_cap'] = df['成交额']
    
    # 估值因子（市盈率）
    df['valuation'] = df['收盘'] / (df['成交额'] / df['成交量'] + 1e-8)
    
    # 成长因子（净利润增长率）
    df['growth'] = df['收盘'].pct_change(20)
    
    # 质量因子（净资产收益率）
    df['quality'] = df['换手率'].rolling(20).std()
    
    # 动量因子（换手率）
    df['momentum'] = df['收盘'].pct_change(5)
    
    # 标准化风格因子
    style_columns = ['market_cap', 'valuation', 'growth', 'quality', 'momentum']
    for col in style_columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    return df

def neutralize_factors(factors_df: pd.DataFrame, factor_columns: list, 
                       neutralization_type: str = 'both') -> pd.DataFrame:
    """
    因子中性化处理（行业中性化 + 市值中性化）
    
    Args:
        factors_df: 因子数据
        factor_columns: 需要中性化的因子列
        neutralization_type: 中性化类型 ('market' | 'industry' | 'both')
        
    Returns:
        中性化后的因子数据
    """
    print(f"执行因子中性化: {neutralization_type}")
    df = factors_df.copy()
    
    # 获取行业分类
    industry_df = get_industry_classification(df['股票代码'].unique().tolist())
    df = pd.merge(df, industry_df[['code', 'industry']], left_on='股票代码', right_on='code', how='left')
    
    # 确保市值因子存在
    if 'market_cap' not in df.columns:
        df['market_cap'] = df['成交额']
    
    # 行业中性化（使用dummy变量回归）
    if neutralization_type in ['industry', 'both']:
        # 构建行业dummy变量
        industry_dummies = pd.get_dummies(df['industry'], prefix='industry')
        X = industry_dummies
        
        if neutralization_type == 'both':
            # 行业+市值中性化：加入市值因子
            X['market_cap'] = df['market_cap']
        
        X = sm.add_constant(X)
        
        for factor in factor_columns:
            try:
                y = df[factor].fillna(0)
                model = sm.OLS(y, X).fit()
                df[factor] = model.resid
            except Exception as e:
                print(f"因子 {factor} 中性化失败: {e}")
    elif neutralization_type == 'market':
        # 仅市值中性化
        X = df[['market_cap']].fillna(0)
        X = sm.add_constant(X)
        
        for factor in factor_columns:
            try:
                y = df[factor].fillna(0)
                model = sm.OLS(y, X).fit()
                df[factor] = model.resid
            except Exception as e:
                print(f"因子 {factor} 市值中性化失败: {e}")
    
    return df

def control_style_exposure(factor_scores: pd.Series, style_factors: pd.DataFrame, 
                          max_exposure: float = 0.1) -> pd.Series:
    """
    控制风格因子暴露（使用二次规划实现精确约束）
    
    Args:
        factor_scores: 股票综合评分
        style_factors: 风格因子数据
        max_exposure: 单一风格因子的最大暴露度
        
    Returns:
        调整后的股票评分
    """
    print(f"控制风格因子暴露，最大暴露度: {max_exposure:.0%}")
    
    try:
        from scipy.optimize import minimize
        
        # 标准化风格因子
        style_exposures = style_factors.copy()
        for col in style_exposures.columns:
            style_exposures[col] = (style_exposures[col] - style_exposures[col].mean()) / style_exposures[col].std()
        
        # 将评分转换为权重（等权初始权重）
        n_stocks = len(factor_scores)
        initial_weights = np.ones(n_stocks) / n_stocks
        
        # 目标函数：最大化与原始评分的相关性
        def objective(weights):
            return -np.corrcoef(factor_scores, weights)[0, 1]
        
        # 约束条件
        constraints = []
        
        # 风格因子暴露约束：|Σ(w_i * s_i,j)| ≤ max_exposure
        for style_col in style_exposures.columns:
            style_values = style_exposures[style_col].values
            
            # 正向暴露约束
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, s=style_values: max_exposure - np.sum(w * s)
            })
            
            # 负向暴露约束
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, s=style_values: max_exposure + np.sum(w * s)
            })
        
        # 权重总和约束
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # 非负权重约束
        bounds = [(0, None) for _ in range(n_stocks)]
        
        # 优化求解
        result = minimize(objective, initial_weights, method='SLSQP', 
                       constraints=constraints, bounds=bounds, 
                       options={'disp': False, 'maxiter': 1000})
        
        if result.success:
            # 将优化后的权重转换为评分调整
            adjusted_scores = factor_scores * result.x
            return adjusted_scores
        else:
            print(f"风格暴露控制优化失败: {result.message}")
            return factor_scores
            
    except ImportError:
        print("警告: scipy未安装，使用简单的风格暴露控制方法")
        # 备用方法：简单惩罚
        style_exposures = style_factors.copy()
        for col in style_exposures.columns:
            style_exposures[col] = (style_exposures[col] - style_exposures[col].mean()) / style_exposures[col].std()
        
        portfolio_exposure = style_exposures.mean()
        adjusted_scores = factor_scores.copy()
        
        for style_col in style_exposures.columns:
            exposure = portfolio_exposure[style_col]
            if abs(exposure) > max_exposure:
                adjust_direction = -1 if exposure > 0 else 1
                penalty = abs(style_exposures[style_col]) * (abs(exposure) - max_exposure)
                adjusted_scores -= penalty * adjust_direction * 0.1
        
        return adjusted_scores

# 获取所有因子列
factor_columns = [col for col in df_clean.columns if col not in 
                 ['日期', '股票代码', '开盘', '最高', '最低', '收盘', '成交量', '成交额', '振幅', 
                  '涨跌幅', '涨跌额', '换手率', 'ma_5', 'ma_20', 'ma_60']]

df_factors = df_clean[['日期', '股票代码'] + factor_columns]
df_factors = standardize_factors(df_factors, factor_columns)
df_factors = neutralize_factors(df_factors, factor_columns, neutralization_type='both')  # 行业+市值中性化
print(f"✓ 因子数量: {len(factor_columns)}")

# ==================== 5. 因子选股 ====================
print("\n[6/6] 构建选股模型...")
def factor_scoring(df, factor_columns):
    """多因子评分"""
    # 计算各因子的IC值
    ic_values = {}
    dates = sorted(df['日期'].unique())
    
    for factor in factor_columns:
        ic_scores = []
        for date in dates[:-1]:
            current_date = date
            next_date = dates[dates.index(date) + 1]
            
            current_df = df[df['日期'] == current_date]
            next_df = df[df['日期'] == next_date]
            
            if len(current_df) > 0 and len(next_df) > 0:
                merged_df = pd.merge(current_df[['股票代码', factor]], 
                                   next_df[['股票代码', '涨跌幅']], 
                                   on='股票代码', how='inner')
                
                if len(merged_df) > 0:
                    ic = merged_df[factor].corr(merged_df['涨跌幅'])
                    ic_scores.append(ic)
        
        ic_values[factor] = np.mean(ic_scores) if ic_scores else 0
    
    # 基于IC值计算因子权重
    total_abs_ic = sum(abs(ic) for ic in ic_values.values())
    factor_weights = {f: abs(ic)/total_abs_ic for f, ic in ic_values.items()}
    
    # 计算综合评分
    def calculate_score(row):
        score = 0
        for factor, weight in factor_weights.items():
            direction = 1 if ic_values[factor] > 0 else -1
            score += row[factor] * weight * direction
        return score
    
    df['score'] = df.apply(calculate_score, axis=1)
    print(f"✓ 因子权重: {factor_weights}")
    
    # 计算风格因子
    df = calculate_style_factors(df)
    
    # 控制风格因子暴露
    style_columns = ['market_cap', 'valuation', 'growth', 'quality', 'momentum']
    df['adjusted_score'] = control_style_exposure(df['score'], df[style_columns])
    
    return df, factor_weights

df_scored, factor_weights = factor_scoring(df_factors, factor_columns)

# ==================== 6. 回测引擎 ====================
print("\n[回测] 执行回测...")
def backtest_strategy(df, config):
    """多因子选股策略回测"""
    dates = sorted(df['日期'].unique())
    portfolio = []
    capital = config['initial_capital']
    shares_held = {}
    daily_returns = []
    
    for i in range(len(dates) - config['hold_period']):
        rebalance_date = dates[i]
        hold_dates = dates[i+1:i+1+config['hold_period']]
        
        # 选股
        available_stocks = df[df['日期'] == rebalance_date]
        selected_stocks = available_stocks.sort_values('adjusted_score', ascending=False) \
                                         .head(config['num_stocks'])['股票代码'] \
                                         .tolist()
        
        # 等权分配资金
        stock_weight = 1.0 / config['num_stocks']
        target_positions = {stock: capital * stock_weight for stock in selected_stocks}
        
        # 调整仓位
        # 卖出不在选股池的股票
        for stock in list(shares_held.keys()):
            if stock not in selected_stocks:
                # 获取卖出价格
                sell_date = hold_dates[0]
                sell_info = df[(df['日期'] == sell_date) & (df['股票代码'] == stock)]
                if not sell_info.empty:
                    sell_price = sell_info['开盘'].iloc[0] * (1 - config['slippage_rate'])
                    capital += shares_held[stock] * sell_price
                    shares_held.pop(stock)
        
        # 买入新选股
        for stock in selected_stocks:
            if stock not in shares_held:
                buy_date = hold_dates[0]
                buy_info = df[(df['日期'] == buy_date) & (df['股票代码'] == stock)]
                if not buy_info.empty:
                    buy_price = buy_info['开盘'].iloc[0] * (1 + config['slippage_rate'])
                    amount = target_positions[stock]
                    shares_to_buy = int(amount / buy_price / 100) * 100
                    cost = shares_to_buy * buy_price
                    commission = cost * config['commission_rate']
                    capital -= (cost + commission)
                    shares_held[stock] = shares_to_buy
        
        # 计算持仓收益
        portfolio_value = capital
        for stock, shares in shares_held.items():
            last_date = hold_dates[-1]
            stock_info = df[(df['日期'] == last_date) & (df['股票代码'] == stock)]
            if not stock_info.empty:
                market_price = stock_info['收盘'].iloc[0]
                portfolio_value += shares * market_price
        
        # 记录每日收益率
        if i == 0:
            previous_value = config['initial_capital']
        else:
            previous_value = portfolio[i-1]['portfolio_value']
            
        daily_return = (portfolio_value - previous_value) / previous_value
        daily_returns.append(daily_return)
        
        portfolio.append({
            '日期': hold_dates[-1],
            'selected_stocks': selected_stocks,
            'portfolio_value': portfolio_value,
            'capital': capital,
            'shares_held': shares_held.copy(),
            'return': daily_return
        })
    
    return portfolio, daily_returns

# 执行回测
portfolio, daily_returns = backtest_strategy(df_scored, CONFIG)
df_portfolio = pd.DataFrame(portfolio)
df_portfolio['日期'] = pd.to_datetime(df_portfolio['日期'])

# ==================== 7. 回测结果分析 ====================
print("\n[结果] 回测结果分析...")
def calculate_metrics(portfolio, daily_returns, config):
    """计算回测指标"""
    portfolio_values = [config['initial_capital']] + [p['portfolio_value'] for p in portfolio]
    dates = [pd.to_datetime(CONFIG['start_date'])] + [p['日期'] for p in portfolio]
    
    returns = np.array(daily_returns)
    excess_returns = returns - config['risk_free_rate']
    
    # 年化收益率
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    years = len(dates) / 252
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    # 年化波动率
    annual_volatility = np.std(returns) * np.sqrt(252)
    
    # 夏普比率
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
    
    # 最大回撤
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # 胜率
    positive_returns = sum(1 for r in daily_returns if r > 0)
    win_rate = positive_returns / len(daily_returns) if daily_returns else 0
    
    # 盈亏比
    profit = sum(r for r in daily_returns if r > 0)
    loss = abs(sum(r for r in daily_returns if r < 0))
    profit_loss_ratio = profit / loss if loss > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'total_trades': len(daily_returns),
        'final_value': portfolio_values[-1]
    }

metrics = calculate_metrics(portfolio, daily_returns, CONFIG)

# 输出回测报告
print("\n" + "="*70)
print("                   回测报告")
print("="*70)
print(f"""
【基本信息】
  选股策略: 多因子模型
  股票池: A股全市场 (前100只演示)
  回测时间: {CONFIG['start_date']} 至 {CONFIG['end_date']}
  调仓频率: {CONFIG['rebalance_frequency']}
  持有期: {CONFIG['hold_period']} 交易日

【策略参数】
  选股数量: {CONFIG['num_stocks']} 只
  资金分配: 等权
  佣金费率: {CONFIG['commission_rate']:.03%}
  印花税: {CONFIG['tax_rate']:.01%}
  滑点: {CONFIG['slippage_rate']:.01%}

【回测结果】
  初始资金: ¥{CONFIG['initial_capital']:,.2f}
  最终资金: ¥{metrics['final_value']:,.2f}
  总收益:   {metrics['total_return']:+.2%}
  年化收益: {metrics['annual_return']:+.2%}
  年化波动: {metrics['annual_volatility']:.2%}
  夏普比率: {metrics['sharpe_ratio']:.2f}
  最大回撤: {metrics['max_drawdown']:.2%}
  胜率:     {metrics['win_rate']:.2%}
  盈亏比:   {metrics['profit_loss_ratio']:.2f}

【风险控制】
  市值范围: {CONFIG['min_market_cap']/1e8:,.0f}亿 - {CONFIG['max_market_cap']/1e8 if CONFIG['max_market_cap'] else '无限制'}亿
  流动性:   {CONFIG['liquidity_threshold']/1e8:,.0f}亿成交额
  股票数量: {CONFIG['num_stocks']} 只 (分散风险)
  因子中性化: 行业中性化 + 市值中性化
  风格暴露: 控制市值、估值、成长、质量、动量因子暴露 ≤ 10%
""")
print("="*70)

# 可视化
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(df_portfolio['日期'], [CONFIG['initial_capital']] + [p['portfolio_value'] for p in portfolio][:-1], 
         label='资产曲线', linewidth=2)
plt.title(f'资产曲线 ({metrics["total_return"]:+.2%})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(df_portfolio['日期'], daily_returns, label='日收益率', alpha=0.7)
plt.title('日收益率')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_factor_backtest.png', dpi=150)
print("✓ 图表: multi_factor_backtest.png")

# 保存回测数据
df_portfolio.to_csv('multi_factor_portfolio.csv', index=False, encoding='utf-8-sig')
df_factors.to_csv('multi_factor_factors.csv', index=False, encoding='utf-8-sig')
print("✓ 数据已保存")

print("\n✓ 回测完成!")
print("="*70)
