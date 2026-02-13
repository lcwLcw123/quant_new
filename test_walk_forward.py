#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-Forward交叉验证框架测试
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
import random
from walk_forward_validation import WalkForwardValidator


def create_test_data(n_days: int = 300, volatility: float = 0.02) -> pd.DataFrame:
    """创建测试数据"""
    dates = []
    base_price = 100
    prices = []
    volume = []
    
    start_date = datetime(2020, 1, 1)
    for i in range(n_days):
        dates.append(start_date + timedelta(days=i))
        base_price += random.gauss(0, base_price * volatility)
        prices.append(base_price)
        volume.append(random.uniform(10000, 1000000))
        
    return pd.DataFrame({
        '日期': dates,
        '开盘': [p * 0.99 for p in prices],
        '最高': [p * 1.02 for p in prices],
        '最低': [p * 0.98 for p in prices],
        '收盘': prices,
        '成交量': volume
    })


def simple_ma_strategy(data: pd.DataFrame, params: dict) -> str:
    """简单均线策略"""
    if len(data) < max(params['ma_short'], params['ma_long']):
        return 'hold'
        
    ma_short = data['收盘'].rolling(params['ma_short']).mean().iloc[-1]
    ma_long = data['收盘'].rolling(params['ma_long']).mean().iloc[-1]
    
    if ma_short > ma_long:
        return 'buy'
    elif ma_short < ma_long:
        return 'sell'
    else:
        return 'hold'


class TestWalkForwardValidator(unittest.TestCase):
    """Walk-Forward验证器测试"""
    
    def test_data_splitting(self):
        """测试数据分割功能"""
        print("测试数据分割功能...")
        data = create_test_data(n_days=600)
        
        validator = WalkForwardValidator(
            training_period=252,
            testing_period=60,
            min_data_points=300
        )
        
        splits = validator.split_data(data)
        self.assertGreater(len(splits), 0)
        print(f"数据分割成功: {len(splits)}个滚动窗口")
        
    def test_parameter_optimization(self):
        """测试参数优化功能"""
        print("测试参数优化功能...")
        data = create_test_data(n_days=300)
        
        validator = WalkForwardValidator(
            training_period=252,
            testing_period=60,
            min_data_points=300
        )
        
        parameter_ranges = {
            'ma_short': [5, 10],
            'ma_long': [20, 30]
        }
        
        best_params = validator.optimize_parameters(
            training_data=data,
            parameter_ranges=parameter_ranges,
            strategy_func=simple_ma_strategy,
            optimize_metric='sharpe_ratio'
        )
        
        self.assertIsNotNone(best_params)
        self.assertIn('ma_short', best_params)
        self.assertIn('ma_long', best_params)
        print(f"参数优化成功: {best_params}")
        
    def test_full_validation(self):
        """测试完整验证流程"""
        print("测试完整验证流程...")
        data = create_test_data(n_days=1000)
        
        validator = WalkForwardValidator(
            training_period=252,
            testing_period=60,
            min_data_points=300
        )
        
        parameter_ranges = {
            'ma_short': [5, 10, 15],
            'ma_long': [30, 40, 50]
        }
        
        results = validator.validate_strategy(
            data=data,
            strategy_func=simple_ma_strategy,
            parameter_ranges=parameter_ranges,
            optimize_metric='sharpe_ratio'
        )
        
        self.assertIsNotNone(results)
        self.assertGreater(len(results['walk_forward_results']), 0)
        print(f"完整验证成功: {len(results['walk_forward_results'])}个周期")
        
    def test_metric_calculation(self):
        """测试指标计算"""
        print("测试指标计算...")
        data = create_test_data(n_days=600)
        
        validator = WalkForwardValidator(
            training_period=252,
            testing_period=60,
            min_data_points=300
        )
        
        parameter_ranges = {
            'ma_short': [5, 10],
            'ma_long': [20, 30]
        }
        
        results = validator.validate_strategy(
            data=data,
            strategy_func=simple_ma_strategy,
            parameter_ranges=parameter_ranges,
            optimize_metric='sharpe_ratio'
        )
        
        metrics = results['risk_metrics']
        self.assertIn('avg_annual_return', metrics)
        self.assertIn('avg_sharpe_ratio', metrics)
        self.assertIn('avg_max_drawdown', metrics)
        
        print(f"平均年化收益率: {metrics['avg_annual_return']*100:.2f}%")
        print(f"平均Sharpe比率: {metrics['avg_sharpe_ratio']:.2f}")
        print(f"平均最大回撤: {metrics['avg_max_drawdown']*100:.2f}%")
        print("指标计算成功")
        
    def test_edge_cases(self):
        """测试边界条件"""
        print("测试边界条件...")
        
        # 测试数据量不足
        data = create_test_data(n_days=200)
        validator = WalkForwardValidator(min_data_points=300)
        
        with self.assertRaises(ValueError):
            validator.split_data(data)
            
        print("数据量不足边界条件测试成功")
        
        # 测试参数范围无效
        data = create_test_data(n_days=400)
        validator = WalkForwardValidator(
            training_period=252,
            testing_period=60,
            min_data_points=300
        )
        
        with self.assertRaises(Exception):
            validator.optimize_parameters(
                training_data=data,
                parameter_ranges={'invalid_param': [1]},
                strategy_func=simple_ma_strategy,
                optimize_metric='invalid_metric'
            )
            
        print("无效参数边界条件测试成功")


if __name__ == "__main__":
    print("=" * 70)
    print("Walk-Forward交叉验证框架测试")
    print("=" * 70)
    print()
    
    unittest.main(verbosity=2)
