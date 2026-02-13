#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的回测引擎模块
提供滑点计算、交易成本、凯利公式仓位、动态止损、移动止损等功能
"""

class BacktestEngine:
    """完善回测引擎"""
    
    def __init__(self, initial_capital=100000):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金，默认10万元
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.side = None  # 用于滑点计算的交易方向
        
    def calculate_slippage(self, price, volume, avg_volume, side='buy'):
        """
        计算滑点 - 基于成交量
        
        成交量越大，滑点越大
        滑点范围：0.05% - 0.5%
        
        Args:
            price: 交易价格
            volume: 成交量
            avg_volume: 平均成交量
            side: 交易方向，'buy' 或 'sell'
            
        Returns:
            float: 调整后的价格
        """
        self.side = side
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        slippage = min(0.005, 0.0005 * (1 + volume_ratio))  # 0.05%-0.5%
        
        if side == 'sell':
            return price * (1 - slippage)
        else:  # buy
            return price * (1 + slippage)
    
    def calculate_transaction_cost(self, price, volume, is_buy=True):
        """
        计算交易成本
        
        包含：
        - 佣金：万二 (0.02%)
        - 印花税：千一 (0.1%)，卖出时收取
        - 冲击成本：约0.1%
        
        Args:
            price: 交易价格
            volume: 交易量
            is_buy: 是否为买入操作
            
        Returns:
            float: 总交易成本
        """
        commission = price * volume * 0.0002  # 万二佣金
        stamp_tax = price * volume * 0.001 if not is_buy else 0  # 印花税（卖出时）
        impact_cost = price * volume * 0.001  # 冲击成本约0.1%
        
        return commission + stamp_tax + impact_cost
    
    def kelly_position(self, win_rate, avg_win, avg_loss):
        """
        凯利公式计算最优仓位
        
        使用半凯利公式，限制在10%-80%之间
        
        Args:
            win_rate: 胜率 (0-1)
            avg_win: 平均盈利幅度
            avg_loss: 平均亏损幅度
            
        Returns:
            float: 最优仓位比例 (0.1-0.8)
        """
        if avg_loss == 0:
            return 0.1
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return min(max(kelly * 0.5, 0.1), 0.8)  # 半凯利，限制10%-80%
    
    def dynamic_stop_loss(self, entry_price, atr):
        """
        动态止损 - 基于ATR
        
        使用2倍ATR作为止损距离
        
        Args:
            entry_price: 入场价格
            atr: 平均真实波幅 (Average True Range)
            
        Returns:
            float: 止损价格
        """
        return entry_price - 2 * atr  # 2倍ATR止损
    
    def trailing_stop(self, current_price, peak_price, callback_pct=0.05):
        """
        移动止损
        
        Args:
            current_price: 当前价格
            peak_price: 最高价/最低价
            callback_pct: 回撤百分比，默认5%
            
        Returns:
            float: 移动止损触发价格
        """
        return peak_price * (1 - callback_pct)
    
    def run_backtest(self, price_data, strategy_func):
        """
        运行回测
        
        Args:
            price_data: 价格数据列表
            strategy_func: 策略函数，接受价格返回交易信号
            
        Returns:
            dict: 回测结果
        """
        self.capital = self.initial_capital
        positions = []
        trades = []
        
        for i, price in enumerate(price_data):
            signal = strategy_func(price, i, price_data)
            
            if signal == 'buy' and not positions:
                # 买入
                volume = self.capital / price
                cost = self.calculate_transaction_cost(price, volume, is_buy=True)
                self.capital -= cost
                positions.append({
                    'entry_price': price,
                    'volume': volume,
                    'entry_idx': i
                })
                trades.append(('BUY', price, volume))
                
            elif signal == 'sell' and positions:
                # 卖出
                position = positions.pop()
                adjusted_price = self.calculate_slippage(price, position['volume'], 
                                                         price_data[i] if i < len(price_data) else price,
                                                         side='sell')
                cost = self.calculate_transaction_cost(adjusted_price, position['volume'], 
                                                        is_buy=False)
                pnl = (adjusted_price * position['volume']) - (position['entry_price'] * position['volume']) - cost
                self.capital += position['volume'] * adjusted_price - cost
                trades.append(('SELL', adjusted_price, position['volume'], pnl))
        
        return {
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'trades': trades,
            'num_trades': len(trades) // 2
        }


# ==================== 演示代码 ====================

if __name__ == "__main__":
    # 示例1: 基础功能演示
    print("=" * 50)
    print("回测引擎功能演示")
    print("=" * 50)
    
    engine = BacktestEngine(initial_capital=100000)
    
    # 测试滑点计算
    print("\n【滑点计算演示】")
    price = 100.0
    volume = 50000
    avg_volume = 20000
    
    buy_price_adjusted = engine.calculate_slippage(price, volume, avg_volume, side='buy')
    sell_price_adjusted = engine.calculate_slippage(price, volume, avg_volume, side='sell')
    
    print(f"原始价格: {price}")
    print(f"买入调整价格: {buy_price_adjusted:.4f} (滑点: {(buy_price_adjusted-price)/price*100:.4f}%)")
    print(f"卖出调整价格: {sell_price_adjusted:.4f} (滑点: {(price-sell_price_adjusted)/price*100:.4f}%)")
    
    # 测试交易成本
    print("\n【交易成本演示】")
    price = 100.0
    volume = 1000
    
    buy_cost = engine.calculate_transaction_cost(price, volume, is_buy=True)
    sell_cost = engine.calculate_transaction_cost(price, volume, is_buy=False)
    
    print(f"买入交易成本: {buy_cost:.2f}元")
    print(f"卖出交易成本: {sell_cost:.2f}元")
    print(f"总成本占比: {(buy_cost + sell_cost) / (price * volume) * 100:.4f}%")
    
    # 测试凯利公式
    print("\n【凯利公式演示】")
    test_cases = [
        (0.5, 0.10, 0.05),   # 50%胜率，盈10%亏5%
        (0.6, 0.15, 0.10),   # 60%胜率，盈15%亏10%
        (0.4, 0.20, 0.10),   # 40%胜率，盈20%亏10%
    ]
    
    for win_rate, avg_win, avg_loss in test_cases:
        position = engine.kelly_position(win_rate, avg_win, avg_loss)
        print(f"胜率{win_rate*100:.0f}%, 盈{avg_win*100:.0f}%, 亏{avg_loss*100:.0f}% → 最优仓位: {position*100:.1f}%")
    
    # 测试止损功能
    print("\n【止损功能演示】")
    entry_price = 100.0
    atr = 2.5
    
    stop_loss = engine.dynamic_stop_loss(entry_price, atr)
    trailing_stop = engine.trailing_stop(95, 105, callback_pct=0.05)
    
    print(f"入场价: {entry_price}, ATR: {atr}")
    print(f"动态止损价: {stop_loss:.2f} (止损幅度: {(entry_price-stop_loss)/entry_price*100:.1f}%)")
    print(f"当前价: 95, 峰值: 105, 移动止损价: {trailing_stop:.2f}")
    
    # 示例2: 简单均线策略回测
    print("\n" + "=" * 50)
    print("简单均线策略回测示例")
    print("=" * 50)
    
    # 从akshare获取真实A股价格数据
    import akshare as ak
    import pandas as pd
    
    try:
        # 获取上证指数数据作为示例
        price_df = ak.stock_zh_a_hist(
            symbol="sh",  # 上证指数
            period="daily",
            start_date="20240101",
            end_date="20240531"
        )
        price_data = price_df['收盘'].tolist()
        print(f"成功获取真实价格数据，共 {len(price_data)} 个交易日")
    except Exception as e:
        print(f"获取真实价格数据失败: {e}")
        print("将使用简化的模拟数据进行演示")
        # 创建简化的模拟数据作为备用方案
        import random
        random.seed(42)
        price_data = []
        base_price = 100
        for i in range(100):
            base_price += random.gauss(0, 2)
            price_data.append(base_price)
    
    # 简单均线策略
    def ma_strategy(price, idx, prices):
        if idx < 20:
            return 'hold'
        ma_short = sum(prices[idx-5:idx]) / 5
        ma_long = sum(prices[idx-20:idx]) / 20
        if ma_short > ma_long:
            return 'buy'
        elif ma_short < ma_long:
            return 'sell'
        return 'hold'
    
    # 运行回测
    engine2 = BacktestEngine(initial_capital=100000)
    results = engine2.run_backtest(price_data, ma_strategy)
    
    print(f"\n初始资金: {100000:.2f}元")
    print(f"最终资金: {results['final_capital']:.2f}元")
    print(f"总收益率: {results['total_return']*100:.2f}%")
    print(f"交易次数: {results['num_trades']}")
    print(f"\n前5笔交易:")
    for i, trade in enumerate(results['trades'][:10]):
        if len(trade) == 4:
            print(f"  {trade[0]}: 价格={trade[1]:.2f}, 数量={trade[2]:.2f}, PnL={trade[3]:.2f}")
        else:
            print(f"  {trade[0]}: 价格={trade[1]:.2f}, 数量={trade[2]:.2f}")
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)
