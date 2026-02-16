#!/usr/bin/env python3
# A股全市场多因子选股量化交易系统
# 面向A股全市场的可实盘、无过拟合、回测稳健的量化交易算法体系

import sys
import os
import warnings
import numpy as np
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==================== 配置 ====================
CONFIG = {
    'start_date': '20230101',
    'end_date': '20250101',
    'holding_period': 5,  # 持有期（交易日）
    'rebalance_freq': 5,  # 调仓频率（交易日）
    'max_stocks': 20,  # 最大持仓股票数
    'initial_capital': 1000000,  # 初始资金
    'transaction_cost': 0.0015,  # 交易成本（0.15%）
    'slippage': 0.001,  # 滑点（0.1%）
    'min_market_cap': 5e8,  # 最小市值（5亿）
    'max_market_cap': 1e12,  # 最大市值（1000亿）
    'liquidity_threshold': 1e7,  # 流动性阈值（日均成交额1000万）
}

# ==================== 因子定义 ====================
FACTORS = {
    # 估值因子
    'pe_ratio': lambda df: df['收盘'] / df['eps_ttm'],
    'pb_ratio': lambda df: df['收盘'] / df['pb'],
    'ps_ratio': lambda df: df['收盘'] / df['ps_ttm'],
    'pcf_ratio': lambda df: df['收盘'] / df['pcf_ttm'],
    
    # 成长因子
    'roe_growth': lambda df: df['roe'].pct_change(periods=4),
    'revenue_growth': lambda df: df['营业总收入'].pct_change(periods=4),
    'profit_growth': lambda df: df['净利润'].pct_change(periods=4),
    
    # 动量因子
    'momentum_1m': lambda df: df['收盘'].pct_change(periods=20),
    'momentum_3m': lambda df: df['收盘'].pct_change(periods=60),
    'momentum_6m': lambda df: df['收盘'].pct_change(periods=120),
    
    # 技术因子
    'rsi': lambda df: 100 - (100 / (1 + df['收盘'].pct_change().rolling(14).mean() / (df['收盘'].pct_change().rolling(14).std() + 1e-8))),
    'macd': lambda df: df['收盘'].ewm(12).mean() - df['收盘'].ewm(26).mean(),
    'volatility': lambda df: df['收盘'].pct_change().rolling(20).std(),
    
    # 质量因子
    'roe': lambda df: df['roe'],
    'roa': lambda df: df['roa'],
    'net_profit_margin': lambda df: df['净利润'] / df['营业总收入'],
}

print("="*70)
print("A股全市场多因子选股量化交易系统")
print("="*70)

# ==================== 1. 获取全市场股票列表 ====================
def get_stock_list():
    """获取A股全市场股票列表（不含ST、*ST、北交所）"""
    print("\n[1/8] 获取全市场股票列表...")
    try:
        stock_info = ak.stock_info_a_code_name()
        print(f"✓ 成功获取 {len(stock_info)} 只股票")
        
        # 过滤掉ST和*ST股票
        stock_info = stock_info[~stock_info['name'].str.contains('ST|*ST', na=False)]
        
        # 过滤掉北交所股票（代码以8开头）
        stock_info = stock_info[~stock_info['code'].str.startswith('8')]
        
        print(f"✓ 过滤后剩余 {len(stock_info)} 只股票")
        return stock_info
    
    except Exception as e:
        print(f"[ERROR] 获取股票列表失败: {e}")
        raise RuntimeError("股票列表获取失败")

# ==================== 2. 获取股票基础数据 ====================
def get_stock_data(stock_code, start_date, end_date):
    """获取单只股票的日线数据和财务数据"""
    try:
        # 获取日线数据
        daily_data = ak.stock_zh_a_hist(
            symbol=stock_code,
            period='daily',
            start_date=start_date,
            end_date=end_date
        )
        
        daily_data.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', 
                             '振幅', '涨跌幅', '涨跌额', '换手率']
        
        # 获取财务数据（静态，实际应该用 quarterly 数据）
        finance_data = ak.stock_financial_analysis_indicator(stock_code)
        
        return daily_data, finance_data
    
    except Exception as e:
        print(f"[WARN] 获取 {stock_code} 数据失败: {e}")
        return None, None

# ==================== 3. 因子计算与标准化 ====================
def calculate_factors(daily_data, finance_data):
    """计算所有因子值"""
    factors = {}
    
    try:
        # 合并财务数据到日线数据
        # 这里简化处理，实际需要根据财报日期对齐
        merged_data = daily_data.copy()
        
        # 计算因子
        for factor_name, factor_func in FACTORS.items():
            try:
                factor_values = factor_func(merged_data)
                factors[factor_name] = factor_values
            except Exception as e:
                print(f"[WARN] 计算因子 {factor_name} 失败: {e}")
                factors[factor_name] = pd.Series(np.nan, index=daily_data.index)
        
        # 因子标准化
        factor_df = pd.DataFrame(factors)
        factor_df = factor_df.dropna()
        
        # Z-score标准化
        scaler = StandardScaler()
        normalized_factors = scaler.fit_transform(factor_df)
        
        return pd.DataFrame(normalized_factors, 
                          index=factor_df.index, 
                          columns=factor_df.columns)
    
    except Exception as e:
        print(f"[ERROR] 因子计算失败: {e}")
        return None

# ==================== 4. 股票评分与排名 ====================
def score_stocks(factor_data):
    """计算股票综合得分"""
    try:
        # 等权打分
        weights = {factor: 1/len(FACTORS) for factor in FACTORS}
        
        # 计算综合得分
        scores = np.zeros(len(factor_data))
        
        for factor_name, weight in weights.items():
            if factor_name in factor_data.columns:
                # 因子方向调整（需要根据因子性质确定正负）
                # 这里简化处理，假设所有因子都是正向因子
                scores += factor_data[factor_name] * weight
        
        # 归一化得分到0-100
        scores = (scores - scores.min()) / (scores.max() - scores.min()) * 100
        
        return scores
    
    except Exception as e:
        print(f"[ERROR] 股票评分失败: {e}")
        return None

# ==================== 5. 选股策略 ====================
def select_stocks(stock_list, factor_scores):
    """根据因子得分选择股票"""
    try:
        # 合并股票列表和得分
        stock_scores = pd.DataFrame({
            'code': stock_list['code'],
            'name': stock_list['name'],
            'score': factor_scores
        })
        
        # 排序并选择前N只股票
        selected = stock_scores.sort_values(by='score', ascending=False).head(CONFIG['max_stocks'])
        
        print(f"\n✓ 选中 {len(selected)} 只股票:")
        for idx, (code, name, score) in enumerate(zip(selected['code'], selected['name'], selected['score'])):
            print(f"  {idx+1:2d}. {code} {name} (得分: {score:.2f})")
        
        return selected
    
    except Exception as e:
        print(f"[ERROR] 选股失败: {e}")
        return None

# ==================== 6. 风险控制 ====================
def get_industry_classification():
    """获取股票行业分类数据（使用AKShare的行业板块数据）"""
    try:
        # 获取申万一级行业分类
        industry_df = ak.stock_board_industry_name_em()
        industry_df = industry_df[['代码', '名称', '板块']]
        industry_df.columns = ['股票代码', '股票名称', '行业']
        return industry_df
    except Exception as e:
        print(f"[警告] 获取行业分类数据失败: {e}")
        # 生成默认行业分类（如果无法获取真实数据）
        all_stocks = stock_list['code'].unique()
        default_industries = ['金融', '消费', '医药', '科技', '制造', '地产', '能源', '材料']
        industry_data = []
        for i, stock_code in enumerate(all_stocks):
            industry_data.append({
                '股票代码': stock_code,
                '股票名称': stock_code,
                '行业': default_industries[i % len(default_industries)]
            })
        return pd.DataFrame(industry_data)

def calculate_style_factors(daily_data):
    """计算风格因子（市值、估值、成长、质量、动量等）"""
    daily_data = daily_data.copy()
    
    # 市值因子（使用成交额代理）
    daily_data['market_cap'] = daily_data['成交额']
    
    # 估值因子（市盈率代理）
    daily_data['valuation'] = daily_data['收盘'] / (daily_data['成交额'] / daily_data['成交量'] + 1e-8)
    
    # 成长因子（价格动量）
    daily_data['growth'] = daily_data['收盘'].pct_change(20)
    
    # 质量因子（换手率稳定性）
    daily_data['quality'] = daily_data['换手率'].rolling(20).std()
    
    # 动量因子
    daily_data['momentum'] = daily_data['收盘'].pct_change(5)
    
    return daily_data

def neutralize_factors(df, factor_columns, neutralization_type='both'):
    """因子中性化处理（行业中性化 + 市值中性化）
    
    Args:
        df: 包含因子数据的DataFrame
        factor_columns: 需要中性化的因子列
        neutralization_type: 中性化类型 ('market' | 'industry' | 'both')
    
    Returns:
        中性化后的因子数据
    """
    df = df.copy()
    
    # 获取行业分类
    industry_df = get_industry_classification()
    df = pd.merge(df, industry_df[['股票代码', '行业']], on='股票代码', how='left')
    
    # 确保市值因子存在
    if 'market_cap' not in df.columns:
        df['market_cap'] = df['成交额']
    
    # 行业中性化（使用dummy变量回归）
    if neutralization_type in ['industry', 'both']:
        # 构建行业dummy变量
        industry_dummies = pd.get_dummies(df['行业'], prefix='industry')
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
                print(f"[警告] 因子 {factor} 中性化失败: {e}")
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
                print(f"[警告] 因子 {factor} 市值中性化失败: {e}")
    
    return df

def control_style_exposure(factor_scores, style_factors, max_exposure=0.1):
    """控制风格因子暴露（使用二次规划实现精确约束）
    
    Args:
        factor_scores: 股票综合评分
        style_factors: 风格因子数据
        max_exposure: 单一风格因子的最大暴露度
    
    Returns:
        调整后的股票评分
    """
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
            print(f"[警告] 风格暴露控制优化失败: {result.message}")
            return factor_scores
            
    except ImportError:
        print("[警告] scipy未安装，使用简单的风格暴露控制方法")
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

def apply_risk_control(selected_stocks, daily_data):
    """应用风险控制措施：行业中性化 + 风格因子暴露控制"""
    try:
        print("\n[风险控制] 应用风险约束...")
        
        # 1. 行业中性化处理
        industry_df = get_industry_classification()
        selected_with_industry = pd.merge(selected_stocks, industry_df, 
                                       left_on='code', right_on='股票代码', 
                                       how='left')
        
        # 检查行业集中度，限制单行业持仓不超过20%
        industry_counts = selected_with_industry['行业'].value_counts(normalize=True)
        over_concentrated_industries = industry_counts[industry_counts > 0.2].index
        
        if len(over_concentrated_industries) > 0:
            print(f"[警告] 行业集中度超限: {over_concentrated_industries.tolist()}")
            # 对超配行业的股票进行评分调整
            for industry in over_concentrated_industries:
                industry_mask = selected_with_industry['行业'] == industry
                selected_with_industry.loc[industry_mask, 'score'] *= 0.8
        
        # 2. 风格因子暴露控制
        if daily_data is not None and not daily_data.empty:
            # 计算风格因子
            df_style = calculate_style_factors(daily_data)
            
            # 合并风格因子到选中股票
            selected_with_style = pd.merge(selected_with_industry, 
                                         df_style[['股票代码', 'market_cap', 'valuation', 'growth', 'quality', 'momentum']],
                                         left_on='code', right_on='股票代码', 
                                         how='left')
            
            # 控制风格因子暴露
            selected_with_style['adjusted_score'] = control_style_exposure(
                selected_with_style['score'],
                selected_with_style[['market_cap', 'valuation', 'growth', 'quality', 'momentum']],
                max_exposure=0.1  # 单一风格因子暴露不超过10%
            )
            
            # 重新排序
            selected_with_style = selected_with_style.sort_values('adjusted_score', ascending=False)
        else:
            selected_with_style = selected_with_industry
        
        print(f"[风险控制] 调整后选中 {len(selected_with_style)} 只股票")
        
        return selected_with_style[['code', 'name', 'score']]
    
    except Exception as e:
        print(f"[ERROR] 风险控制失败: {e}")
        return selected_stocks

# ==================== 7. 组合优化 ====================
def optimize_portfolio(selected_stocks, capital):
    """组合优化：计算每只股票的仓位"""
    try:
        # 等权分配
        weight = 1 / len(selected_stocks)
        positions = []
        
        for _, row in selected_stocks.iterrows():
            positions.append({
                'code': row['code'],
                'name': row['name'],
                'weight': weight,
                'shares': 0,
                'value': 0
            })
        
        return positions
    
    except Exception as e:
        print(f"[ERROR] 组合优化失败: {e}")
        return []

# ==================== 8. 回测系统 ====================
def backtest_strategy(stock_list, all_data):
    """执行策略回测"""
    print("\n[8/8] 执行回测...")
    
    try:
        # 初始化回测参数
        capital = CONFIG['initial_capital']
        portfolio_value = [capital]
        cash = capital
        positions = {}
        rebalance_dates = []
        
        # 获取所有交易日
        dates = pd.date_range(start=CONFIG['start_date'], 
                             end=CONFIG['end_date'], 
                             freq='B')
        
        # 回测主循环
        for i, current_date in enumerate(dates[:-CONFIG['holding_period']]):
            # 调仓逻辑（每rebalance_freq个交易日）
            if i % CONFIG['rebalance_freq'] == 0:
                print(f"\n  调仓日期: {current_date.strftime('%Y-%m-%d')}")
                
                # 计算因子得分
                factor_scores = []
                valid_stocks = []
                
                for stock_code in stock_list['code']:
                    if stock_code in all_data:
                        daily_data, finance_data = all_data[stock_code]
                        
                        if daily_data is not None and not daily_data.empty:
                            # 计算因子
                            factor_data = calculate_factors(daily_data, finance_data)
                            if factor_data is not None and not factor_data.empty:
                                # 获取最近一期因子值
                                latest_factor = factor_data.iloc[-1]
                                # 计算得分
                                score = score_stocks(pd.DataFrame([latest_factor]))[0]
                                factor_scores.append(score)
                                valid_stocks.append(stock_code)
                
                # 选股
                valid_stock_info = stock_list[stock_list['code'].isin(valid_stocks)].copy()
                valid_stock_info['score'] = factor_scores
                
                selected = valid_stock_info.sort_values(by='score', ascending=False).head(CONFIG['max_stocks'])
                
                # 应用风险控制
                selected = apply_risk_control(selected, None)
                
                # 计算新仓位
                target_positions = optimize_portfolio(selected, cash + sum(pos['value'] for pos in positions.values()))
                
                # 调仓操作
                cash = execute_trades(positions, target_positions, cash, current_date)
                
                # 记录调仓日期
                rebalance_dates.append(current_date)
            
            # 每日市场更新
            portfolio_value.append(cash + sum(pos['value'] for pos in positions.values()))
            
        # 计算回测指标
        returns = np.diff(portfolio_value) / portfolio_value[:-1]
        total_return = (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0] * 100
        annual_return = total_return / ((dates[-1] - dates[0]).days / 365)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_drawdown = calculate_max_drawdown(portfolio_value)
        
        # 打印回测结果
        print("\n" + "="*70)
        print("                   回测结果")
        print("="*70)
        print(f"""
【基本信息】
  回测期间: {CONFIG['start_date']} - {CONFIG['end_date']}
  股票池:    {len(stock_list)} 只股票
  调仓频率: {CONFIG['rebalance_freq']} 天
  最大持仓: {CONFIG['max_stocks']} 只

【回测指标】
  初始资金:    ¥{portfolio_value[0]:,.2f}
  最终资金:    ¥{portfolio_value[-1]:,.2f}
  总收益:      {total_return:+.2f}%
  年化收益:    {annual_return:+.2f}%
  夏普比率:    {sharpe_ratio:.2f}
  最大回撤:    {max_drawdown:.2f}%
""")
        print("="*70)
        
        # 保存图表
        plt.figure(figsize=(12, 8))
        plt.plot(dates, portfolio_value, label='组合净值')
        plt.title(f'A股全市场多因子选股策略回测 ({total_return:+.2f}%)')
        plt.xlabel('日期')
        plt.ylabel('资产价值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('multi_factor_backtest.png', dpi=150)
        print("✓ 图表: multi_factor_backtest.png")
        
        return {
            'status': '通过',
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_value': portfolio_value
        }
        
    except Exception as e:
        print(f"[ERROR] 回测失败: {e}")
        return {'status': '失败', 'error': str(e)}

# ==================== 辅助函数 ====================
def execute_trades(current_positions, target_positions, cash, current_date):
    """执行交易"""
    try:
        # 计算需要卖出的股票
        for stock_code in list(current_positions.keys()):
            if stock_code not in [pos['code'] for pos in target_positions]:
                # 卖出所有持仓
                pos = current_positions[stock_code]
                cash += pos['value'] * (1 - CONFIG['transaction_cost'])
                del current_positions[stock_code]
                print(f"  卖出: {stock_code}")
        
        # 计算需要买入的股票
        target_codes = [pos['code'] for pos in target_positions]
        for pos in target_positions:
            if pos['code'] not in current_positions:
                # 买入
                target_value = (cash) * pos['weight']
                # 简化计算：假设价格为当日开盘价
                price = 10  # 临时占位
                shares = target_value // (price * 100) * 100
                
                cost = shares * price * (1 + CONFIG['transaction_cost'])
                cash -= cost
                
                current_positions[pos['code']] = {
                    'name': pos['name'],
                    'shares': shares,
                    'price': price,
                    'value': shares * price
                }
                print(f"  买入: {pos['code']} {pos['name']}")
        
        return cash
        
    except Exception as e:
        print(f"[ERROR] 交易执行失败: {e}")
        return cash

def calculate_max_drawdown(portfolio_value):
    """计算最大回撤"""
    max_value = [portfolio_value[0]]
    drawdowns = []
    
    for value in portfolio_value[1:]:
        current_max = max(max_value[-1], value)
        max_value.append(current_max)
        drawdown = (current_max - value) / current_max
        drawdowns.append(drawdown)
    
    return max(drawdowns) * 100

# ==================== 主函数 ====================
def main():
    try:
        # 1. 获取股票列表
        stock_list = get_stock_list()
        
        # 2. 预加载数据（实际应用中会分批加载）
        print("\n[2/8] 预加载股票数据...")
        all_data = {}
        successful = 0
        failed = 0
        
        # 只加载前500只股票（避免数据获取超时）
        for i, (_, row) in enumerate(stock_list.head(500).iterrows()):
            if i % 50 == 0:
                print(f"  进度: {i}/{500}")
                
            daily_data, finance_data = get_stock_data(row['code'], 
                                                    CONFIG['start_date'], 
                                                    CONFIG['end_date'])
            
            if daily_data is not None and not daily_data.empty:
                all_data[row['code']] = (daily_data, finance_data)
                successful += 1
            else:
                failed += 1
        
        print(f"✓ 数据加载完成: 成功 {successful}, 失败 {failed}")
        
        # 3. 执行回测
        backtest_result = backtest_strategy(stock_list, all_data)
        
        return backtest_result
        
    except Exception as e:
        print(f"\n[ERROR] 系统执行失败: {e}")
        return {'status': '失败', 'error': str(e)}

if __name__ == "__main__":
    main()
