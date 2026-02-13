
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward 交叉验证框架
功能：
- 实现滚动窗口交叉验证，防止过拟合
- 支持训练期和测试期长度配置
- 计算样本外预测结果和性能指标
- 提供可视化分析功能
- 与现有回测引擎兼容

Walk-Forward 交叉验证将整个回测期分为多个滚动窗口，每个窗口分为训练期和测试期。
通过在训练期优化参数，在测试期验证，有效防止过拟合。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Callable, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from vectorized_backtest import VectorizedBacktestEngine, VectorizedResult
from backtest_engine import EnhancedBacktestEngine, TradeCostConfig

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
logger.addHandler(console_handler)


@dataclass
class WalkForwardConfig:
    """Walk-Forward 交叉验证配置"""
    train_length: int = 242  # 训练期长度（交易日），默认1年（242个交易日）
    test_length: int = 60    # 测试期长度（交易日），默认2个月（约60个交易日）
    min_train_length: int = 120  # 最小训练期长度
    step_size: int = 30      # 滚动步长（交易日），默认1个月
    initial_capital: float = 1000000.0  # 初始资金
    transaction_cost: float = 0.0015  # 交易成本
    slippage: float = 0.001  # 滑点
    risk_free_rate: float = 0.03  # 无风险利率
    trading_days_per_year: int = 242  # 年交易日数量
    plot_results: bool = True  # 是否绘制结果图表
    random_seed: int = 42  # 随机种子，用于可复现性

    def validate(self) -> None:
        """验证配置参数"""
        errors = []
        if self.train_length <= 0:
            errors.append(f"train_length 必须为正数，当前值: {self.train_length}")
        if self.test_length <= 0:
            errors.append(f"test_length 必须为正数，当前值: {self.test_length}")
        if self.min_train_length <= 0 or self.min_train_length > self.train_length:
            errors.append(f"min_train_length 必须在(0, {self.train_length}]范围内，当前值: {self.min_train_length}")
        if self.step_size <= 0:
            errors.append(f"step_size 必须为正数，当前值: {self.step_size}")
        if self.initial_capital <= 0:
            errors.append(f"initial_capital 必须为正数，当前值: {self.initial_capital}")
        if self.transaction_cost < 0 or self.transaction_cost > 0.1:
            errors.append(f"transaction_cost 必须在[0, 0.1]范围内，当前值: {self.transaction_cost}")
        if self.slippage < 0 or self.slippage > 0.1:
            errors.append(f"slippage 必须在[0, 0.1]范围内，当前值: {self.slippage}")
        if self.risk_free_rate < 0 or self.risk_free_rate > 0.2:
            errors.append(f"risk_free_rate 必须在[0, 0.2]范围内，当前值: {self.risk_free_rate}")
        if errors:
            raise ValueError("\n".join(errors))


class WalkForwardValidator:
    """Walk-Forward 交叉验证器"""
    
    def __init__(self, config: WalkForwardConfig = None):
        """
        初始化 Walk-Forward 验证器
        
        Args:
            config: 验证配置
        """
        self.config = config or WalkForwardConfig()
        self.config.validate()
        self.logger = logger
        self.results = []
        self.total_equity_curve = []
        self.total_trades = []
        self.best_params = []
        
    def split_data(self, prices: pd.Series, signals: Optional[pd.Series] = None) -> List[Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """
        将时间序列数据分割为训练期和测试期的滚动窗口
        
        Args:
            prices: 价格序列
            signals: 信号序列（可选）
            
        Returns:
            训练期价格、训练期信号、测试期价格、测试期信号的元组列表
        """
        n = len(prices)
        splits = []
        
        start = 0
        while start + self.config.train_length + self.config.test_length <= n:
            # 训练期
            train_end = start + self.config.train_length
            train_prices = prices.iloc[start:train_end]
            train_signals = signals.iloc[start:train_end] if signals is not None else None
            
            # 测试期
            test_start = train_end
            test_end = test_start + self.config.test_length
            test_prices = prices.iloc[test_start:test_end]
            test_signals = signals.iloc[test_start:test_end] if signals is not None else None
            
            splits.append((train_prices, train_signals, test_prices, test_signals))
            
            start += self.config.step_size
        
        self.logger.info(f"数据分割完成，共生成 {len(splits)} 个滚动窗口")
        return splits
    
    def optimize_parameters(self, train_prices: pd.Series, signal_fn, param_grid: Dict[str, List]) -> Dict:
        """
        在训练期上优化策略参数
        
        Args:
            train_prices: 训练期价格数据
            signal_fn: 信号生成函数，签名: signal_fn(prices, **params) -> pd.Series
            param_grid: 参数网格，如 {"fast": [5,10], "slow": [30,60]}
            
        Returns:
            最佳参数组合
        """
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))
        
        self.logger.info(f"参数优化开始，共 {len(combos)} 组参数")
        
        best_score = -np.inf
        best_params = None
        
        # 创建回测引擎
        cost_config = TradeCostConfig(
            commission_rate=self.config.transaction_cost,
            stamp_tax_rate=0.001,
            slippage_base=self.config.slippage,
            min_commission=5.0
        )
        
        backtester = VectorizedBacktestEngine(
            initial_capital=self.config.initial_capital,
            trade_cost_config=cost_config,
            risk_free_rate=self.config.risk_free_rate,
            trading_days_per_year=self.config.trading_days_per_year
        )
        
        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                # 生成训练期信号
                signals = signal_fn(train_prices, **params)
                
                # 运行回测
                result = backtester.run(train_prices, signals)
                
                # 使用夏普比率作为优化指标
                if result.sharpe_ratio > best_score:
                    best_score = result.sharpe_ratio
                    best_params = params
                    
                self.logger.debug(f"参数组合 {params} -> 夏普比率: {result.sharpe_ratio:.4f}")
                
            except Exception as e:
                self.logger.warning(f"参数组合 {params} 优化失败: {e}")
        
        self.logger.info(f"参数优化完成，最佳参数: {best_params}，最佳夏普比率: {best_score:.4f}")
        return best_params
    
    def run_validation(self, prices: pd.Series, signal_fn, param_grid: Dict[str, List]) -> Dict:
        """
        运行完整的 Walk-Forward 验证
        
        Args:
            prices: 完整价格序列
            signal_fn: 信号生成函数，签名: signal_fn(prices, **params) -> pd.Series
            param_grid: 参数网格，用于参数优化
            
        Returns:
            验证结果总结
        """
        self.logger.info("=" * 70)
        self.logger.info("开始 Walk-Forward 交叉验证")
        self.logger.info("=" * 70)
        
        # 分割数据为滚动窗口
        splits = self.split_data(prices)
        
        if len(splits) == 0:
            raise ValueError("数据长度不足以进行 Walk-Forward 验证")
        
        total_equity = []
        total_trades = []
        window_metrics = []
        
        for i, (train_prices, _, test_prices, _) in enumerate(splits):
            self.logger.info(f"处理窗口 {i+1}/{len(splits)}")
            
            # 在训练期上优化参数
            best_params = self.optimize_parameters(train_prices, signal_fn, param_grid)
            self.best_params.append(best_params)
            
            # 在测试期上测试
            test_signals = signal_fn(test_prices, **best_params)
            
            # 创建回测引擎
            cost_config = TradeCostConfig(
                commission_rate=self.config.transaction_cost,
                stamp_tax_rate=0.001,
                slippage_base=self.config.slippage,
                min_commission=5.0
            )
            
            backtester = VectorizedBacktestEngine(
                initial_capital=self.config.initial_capital,
                trade_cost_config=cost_config,
                risk_free_rate=self.config.risk_free_rate,
                trading_days_per_year=self.config.trading_days_per_year
            )
            
            test_result = backtester.run(test_prices, test_signals)
            
            # 保存结果
            self.results.append({
                'window': i,
                'train_start': train_prices.index[0],
                'train_end': train_prices.index[-1],
                'test_start': test_prices.index[0],
                'test_end': test_prices.index[-1],
                'best_params': best_params,
                'result': test_result,
            })
            
            window_metrics.append({
                'window': i,
                'total_return': test_result.total_return,
                'annualized_return': test_result.annualized_return,
                'sharpe_ratio': test_result.sharpe_ratio,
                'max_drawdown': test_result.max_drawdown,
                'calmar_ratio': test_result.calmar_ratio,
                'total_trades': test_result.total_trades,
                'win_rate': test_result.win_rate,
                'profit_loss_ratio': test_result.profit_loss_ratio,
                'total_cost': test_result.total_cost,
            })
            
            # 计算累计权益曲线
            if total_equity:
                # 计算测试期的权益曲线相对于前一窗口的增长
                test_equity = total_equity[-1] * (1 + test_result.total_return)
                total_equity.append(test_equity)
            else:
                total_equity.append(self.config.initial_capital * (1 + test_result.total_return))
                
            total_trades.append(test_result.total_trades)
        
        # 计算整体指标
        summary = self._calculate_summary(window_metrics, total_equity)
        
        # 绘制结果
        if self.config.plot_results:
            self._plot_results(total_equity, window_metrics)
            
        return summary
    
    def _calculate_summary(self, window_metrics: List[Dict], total_equity: List[float]) -> Dict:
        """
        计算 Walk-Forward 验证的整体统计指标
        
        Args:
            window_metrics: 各个窗口的指标
            total_equity: 总权益曲线
            
        Returns:
            整体统计指标
        """
        metrics_df = pd.DataFrame(window_metrics)
        
        # 计算整体收益率
        total_return = (total_equity[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # 计算年化收益率
        n_windows = len(total_equity)
        total_days = n_windows * self.config.test_length
        annual_return = (1 + total_return) ** (self.config.trading_days_per_year / total_days) - 1
        
        # 计算年化波动率
        daily_returns = []
        for window in self.results:
            daily_returns.extend(window['result'].daily_returns.tolist())
            
        annual_volatility = np.std(daily_returns) * np.sqrt(self.config.trading_days_per_year)
        
        # 计算夏普比率
        sharpe_ratio = (annual_return - self.config.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 计算最大回撤
        max_drawdown = 0.0
        peak = self.config.initial_capital
        
        for window in self.results:
            equity = window['result'].equity_curve
            for val in equity:
                current = val
                if current > peak:
                    peak = current
                drawdown = (current - peak) / peak
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
        
        summary = {
            'total_windows': len(self.results),
            'total_return': total_return,
            'annualized_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'annual_volatility': annual_volatility,
            'max_drawdown': max_drawdown,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'avg_total_return_per_window': metrics_df['total_return'].mean(),
            'avg_sharpe_per_window': metrics_df['sharpe_ratio'].mean(),
            'avg_max_drawdown_per_window': metrics_df['max_drawdown'].mean(),
            'avg_total_trades_per_window': metrics_df['total_trades'].mean(),
            'avg_win_rate_per_window': metrics_df['win_rate'].mean(),
            'avg_cost_per_window': metrics_df['total_cost'].mean(),
            'best_params_stats': self._analyze_best_params(),
            'window_metrics': window_metrics,
            'total_equity_curve': total_equity,
            'total_trades': sum(metrics_df['total_trades']),
        }
        
        self.logger.info("=" * 70)
        self.logger.info("Walk-Forward 验证完成")
        self.logger.info("=" * 70)
        
        return summary
    
    def _analyze_best_params(self) -> Dict:
        """
        分析最佳参数的统计特征
        
        Returns:
            参数统计信息
        """
        params_df = pd.DataFrame(self.best_params)
        param_stats = {}
        
        for col in params_df.columns:
            param_stats[col] = {
                'min': params_df[col].min(),
                'max': params_df[col].max(),
                'mean': params_df[col].mean(),
                'std': params_df[col].std(),
                'counts': params_df[col].value_counts().to_dict()
            }
            
        return param_stats
    
    def _plot_results(self, total_equity: List[float], window_metrics: List[Dict]) -> None:
        """
        绘制 Walk-Forward 验证结果的可视化图表
        
        Args:
            total_equity: 总权益曲线
            window_metrics: 各个窗口的指标
        """
        plt.figure(figsize=(15, 12))
        
        # 1. 累计权益曲线
        plt.subplot(2, 2, 1)
        plt.plot(total_equity, 'b-', linewidth=2)
        plt.title('Walk-Forward 验证累计权益曲线')
        plt.ylabel('资产价值 (元)')
        plt.grid(True, alpha=0.3)
        
        # 2. 窗口夏普比率
        plt.subplot(2, 2, 2)
        sharpe_values = [wm['sharpe_ratio'] for wm in window_metrics]
        plt.bar(range(len(sharpe_values)), sharpe_values, color='g')
        plt.title('每个窗口的夏普比率')
        plt.xlabel('窗口索引')
        plt.ylabel('夏普比率')
        plt.grid(True, alpha=0.3)
        
        # 3. 窗口最大回撤
        plt.subplot(2, 2, 3)
        drawdown_values = [wm['max_drawdown'] for wm in window_metrics]
        plt.bar(range(len(drawdown_values)), drawdown_values, color='r')
        plt.title('每个窗口的最大回撤')
        plt.xlabel('窗口索引')
        plt.ylabel('最大回撤')
        plt.grid(True, alpha=0.3)
        
        # 4. 窗口交易次数
        plt.subplot(2, 2, 4)
        trade_counts = [wm['total_trades'] for wm in window_metrics]
        plt.plot(trade_counts, 'k-', linewidth=2)
        plt.title('每个窗口的交易次数')
        plt.xlabel('窗口索引')
        plt.ylabel('交易次数')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('walk_forward_validation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("结果图表已保存为 walk_forward_validation_results.png")
    
    def print_summary(self, summary: Dict) -> None:
        """
        打印 Walk-Forward 验证的详细报告
        
        Args:
            summary: 验证结果总结
        """
        print("\n" + "=" * 70)
        print("                 Walk-Forward 验证报告")
        print("=" * 70)
        
        print(f"""
【整体性能】
  窗口数量:             {summary['total_windows']}
  总收益率:             {summary['total_return']*100:.2f}%
  年化收益率:           {summary['annualized_return']*100:.2f}%
  年化波动率:           {summary['annual_volatility']*100:.2f}%
  夏普比率:             {summary['sharpe_ratio']:.2f}
  最大回撤:             {summary['max_drawdown']*100:.2f}%
  卡尔马比率:           {summary['calmar_ratio']:.2f}

【交易统计】
  总交易次数:           {summary['total_trades']}
  平均窗口交易次数:     {summary['avg_total_trades_per_window']:.1f}
  平均窗口胜率:         {summary['avg_win_rate_per_window']*100:.2f}%

【成本统计】
  平均窗口交易成本:     ¥{summary['avg_cost_per_window']:.2f}

【参数统计】
  参数分布:
""")
        
        for param, stats in summary['best_params_stats'].items():
            print(f"    {param}:")
            print(f"      最小值: {stats['min']:.1f}")
            print(f"      最大值: {stats['max']:.1f}")
            print(f"      平均值: {stats['mean']:.1f}")
            print(f"      标准差: {stats['std']:.1f}")
            print(f"      分布: {dict(stats['counts'])}")
            print()
        
        print(f"""
【窗口统计】
  窗口夏普比率:         {summary['avg_sharpe_per_window']:.2f} ± {np.std([wm['sharpe_ratio'] for wm in summary['window_metrics']]):.2f}
  窗口最大回撤:         {summary['avg_max_drawdown_per_window']*100:.2f}% ± {np.std([wm['max_drawdown'] for wm in summary['window_metrics']])*100:.2f}%
""")
        
        print("=" * 70)
    
    def detect_overfitting(self, summary: Dict, significance_level: float = 0.05) -> Dict:
        """
        检测策略的过拟合风险
        
        Args:
            summary: 验证结果
            significance_level: 显著性水平（默认0.05）
            
        Returns:
            过拟合检测结果
        """
        self.logger.info("开始检测过拟合风险")
        
        metrics_df = pd.DataFrame(summary['window_metrics'])
        
        # 计算训练期和测试期的统计差异
        training_performance = []
        testing_performance = []
        
        for window in self.results:
            # 在训练期上评估参数
            cost_config = TradeCostConfig(
                commission_rate=self.config.transaction_cost,
                stamp_tax_rate=0.001,
                slippage_base=self.config.slippage,
                min_commission=5.0
            )
            
            backtester = VectorizedBacktestEngine(
                initial_capital=self.config.initial_capital,
                trade_cost_config=cost_config,
                risk_free_rate=self.config.risk_free_rate,
                trading_days_per_year=self.config.trading_days_per_year
            )
            
            # 重新在训练期上测试最佳参数
            params = window['best_params']
            
            # 重新获取训练期数据
            train_prices = window['result'].equity_curve.index  # 这里需要修改，实际应该重新获取训练期数据
            
            # 生成训练期信号
            train_signals = self.results[window['window']]['signal_fn'](train_prices, **params)
            
            # 运行训练期回测
            train_result = backtester.run(train_prices, train_signals)
            
            training_performance.append(train_result.sharpe_ratio)
            testing_performance.append(window['result'].sharpe_ratio)
        
        # 使用 t 检验比较训练期和测试期的性能
        from scipy import stats
        
        t_stat, p_value = stats.ttest_rel(training_performance, testing_performance)
        
        overfitting_risk = '高' if p_value < significance_level and np.mean(training_performance) > np.mean(testing_performance) else '低'
        
        self.logger.info(f"过拟合风险评估完成，风险级别: {overfitting_risk}")
        
        return {
            'risk_level': overfitting_risk,
            'p_value': p_value,
            'significance_level': significance_level,
            'training_performance': training_performance,
            'testing_performance': testing_performance,
            'training_mean': np.mean(training_performance),
            'testing_mean': np.mean(testing_performance),
            't_statistic': t_stat
        }


# ==================== 演示代码 ====================
if __name__ == "__main__":
    import akshare as ak
    
    # 设置日志级别为 DEBUG，以便查看详细信息
    logger.setLevel(logging.DEBUG)
    
    try:
        # 从 AKShare 获取上证指数数据作为示例
        price_df = ak.stock_zh_a_hist(
            symbol="sh",
            period="daily",
            start_date="20200101",
            end_date="20241231"
        )
        
        price_df['日期'] = pd.to_datetime(price_df['日期'])
        price_data = price_df.set_index('日期')['收盘']
        
        print("=" * 70)
        print("          Walk-Forward 验证演示")
        print("=" * 70)
        print(f"获取到 {len(price_data)} 个交易日数据")
        print(f"日期范围: {price_data.index[0]} 到 {price_data.index[-1]}")
        print()
        
        # 简单的均线交叉策略信号生成函数
        def ma_strategy(prices, fast_period, slow_period):
            """
            简单的均线交叉策略
            
            Args:
                prices: 价格序列
                fast_period: 短期均线周期
                slow_period: 长期均线周期
                
            Returns:
                交易信号序列（1=买入，-1=卖出，0=持有）
            """
            ma_fast = prices.rolling(window=fast_period).mean()
            ma_slow = prices.rolling(window=slow_period).mean()
            
            signals = pd.Series(0, index=prices.index)
            
            # 买入信号：短期均线上穿长期均线
            signals[ma_fast > ma_slow] = 1
            # 卖出信号：短期均线下穿长期均线
            signals[ma_fast < ma_slow] = -1
            
            return signals
        
        # 参数网格
        param_grid = {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [30, 40, 50, 60]
        }
        
        # 创建 Walk-Forward 验证器
        config = WalkForwardConfig(
            train_length=242,  # 1年训练期
            test_length=60,    # 2个月测试期
            step_size=30,      # 1个月滚动步长
            initial_capital=1000000,
            transaction_cost=0.0015,
            slippage=0.001,
            plot_results=True
        )
        
        validator = WalkForwardValidator(config)
        
        # 运行验证
        summary = validator.run_validation(price_data, ma_strategy, param_grid)
        
        # 打印报告
        validator.print_summary(summary)
        
        # 检测过拟合风险
        overfitting_result = validator.detect_overfitting(summary)
        
        print(f"\n【过拟合检测】")
        print(f"  风险级别: {overfitting_result['risk_level']}")
        print(f"  显著性水平: {overfitting_result['significance_level']:.2f}")
        print(f"  p值: {overfitting_result['p_value']:.4f}")
        print(f"  训练期性能: {overfitting_result['training_mean']:.2f}")
        print(f"  测试期性能: {overfitting_result['testing_mean']:.2f}")
        
        print("\n✓ Walk-Forward 验证演示完成！")
        print(f"结果图表已保存为 walk_forward_validation_results.png")
        
    except Exception as e:
        logger.error(f"演示失败: {e}")
        print(f"❌ 程序执行失败: {e}")
