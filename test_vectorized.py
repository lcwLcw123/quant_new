#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量化回测引擎测试"""

import sys
sys.path.insert(0, "/Users/hanyinghui/quant_new")


def test_syntax():
    """语法检查"""
    import py_compile
    py_compile.compile("/Users/hanyinghui/quant_new/vectorized_backtest.py", doraise=True)
    print("✅ 语法检查通过")


def test_import():
    """导入测试"""
    from vectorized_backtest import VectorizedBacktestEngine, VectorizedResult
    print("✅ 导入测试通过")


def test_basic_backtest():
    """基本回测功能测试"""
    import numpy as np
    import pandas as pd
    from vectorized_backtest import VectorizedBacktestEngine

    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    # 模拟价格：随机游走
    returns = np.random.normal(0.001, 0.02, 100)
    prices = pd.Series(10 * np.exp(np.cumsum(returns)), index=dates)

    # 简单信号：均线金叉/死叉
    ma_fast = prices.rolling(5).mean()
    ma_slow = prices.rolling(20).mean()
    signals = pd.Series(0, index=dates)
    signals[ma_fast > ma_slow] = 1
    signals[ma_fast <= ma_slow] = -1
    signals = signals.fillna(0).astype(int)

    engine = VectorizedBacktestEngine(initial_capital=100000)
    result = engine.run(prices, signals)

    assert result.equity_curve is not None, "equity_curve 不应为 None"
    assert len(result.equity_curve) == len(prices), "equity_curve 长度应与 prices 一致"
    assert result.total_trades >= 0, "交易次数不应为负"
    assert -1 <= result.max_drawdown <= 0, "最大回撤应在 [-1, 0] 范围"

    print(f"✅ 基本回测通过")
    for k, v in result.summary().items():
        print(f"   {k}: {v}")


def test_no_trades():
    """无交易场景"""
    import numpy as np
    import pandas as pd
    from vectorized_backtest import VectorizedBacktestEngine

    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    prices = pd.Series(np.linspace(10, 15, 50), index=dates)
    signals = pd.Series(0, index=dates)  # 全空仓

    engine = VectorizedBacktestEngine(initial_capital=50000)
    result = engine.run(prices, signals)

    assert result.total_trades == 0, "应无交易"
    assert abs(result.total_return) < 1e-6, "空仓应无收益"
    print("✅ 无交易场景通过")


def test_parameter_sweep():
    """参数扫描测试"""
    import numpy as np
    import pandas as pd
    from vectorized_backtest import VectorizedBacktestEngine

    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=200, freq="B")
    returns = np.random.normal(0.001, 0.02, 200)
    prices = pd.Series(10 * np.exp(np.cumsum(returns)), index=dates)

    def ma_signal(prices, fast=5, slow=20):
        ma_f = prices.rolling(fast).mean()
        ma_s = prices.rolling(slow).mean()
        sig = pd.Series(0, index=prices.index)
        sig[ma_f > ma_s] = 1
        sig[ma_f <= ma_s] = -1
        return sig.fillna(0).astype(int)

    engine = VectorizedBacktestEngine(initial_capital=100000)
    result_df = engine.parameter_sweep(
        prices,
        ma_signal,
        {"fast": [5, 10], "slow": [20, 40]},
        metric="sharpe_ratio",
    )

    assert len(result_df) == 4, f"应有 4 组结果，实际 {len(result_df)}"
    assert "sharpe_ratio" in result_df.columns
    print(f"✅ 参数扫描通过，最佳夏普: {result_df.iloc[0]['sharpe_ratio']:.2f}")
    print(result_df[["fast", "slow", "sharpe_ratio", "total_return"]].to_string(index=False))


def test_validation():
    """输入验证测试"""
    import pandas as pd
    from vectorized_backtest import VectorizedBacktestEngine, InputValidationError

    engine = VectorizedBacktestEngine()

    # 空数据
    try:
        engine.run(pd.Series(dtype=float), pd.Series(dtype=float))
        assert False, "应抛出异常"
    except InputValidationError:
        print("✅ 空数据验证通过")

    # 负初始资金
    try:
        VectorizedBacktestEngine(initial_capital=-1000)
        assert False, "应抛出异常"
    except InputValidationError:
        print("✅ 负初始资金验证通过")


if __name__ == "__main__":
    test_syntax()
    test_import()
    test_basic_backtest()
    test_no_trades()
    test_parameter_sweep()
    test_validation()
    print("\n🎉 全部测试通过!")
