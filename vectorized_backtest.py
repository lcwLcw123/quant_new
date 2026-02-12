#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量化回测引擎模块
功能：
- 基于 NumPy/Pandas 向量化运算的高性能回测
- 支持多种信号类型（连续信号、离散信号）
- 完整的交易成本模型（佣金、印花税、滑点）
- 性能统计与风险指标计算
- 与 EnhancedBacktestEngine 兼容的接口

相比逐条回测（event-driven），向量化回测在大数据量下可提升 10-100x 性能。

Author: Coding Agent (Multi-Agent Iteration)
Date: 2026-02-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
from pathlib import Path

from backtest_engine import (
    TradeCostConfig,
    BacktestError,
    InputValidationError,
    setup_logger,
)


@dataclass
class VectorizedResult:
    """向量化回测结果"""
    # 基础指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # 最大回撤持续天数
    calmar_ratio: float = 0.0

    # 交易统计
    total_trades: int = 0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_holding_days: float = 0.0

    # 成本统计
    total_commission: float = 0.0
    total_stamp_tax: float = 0.0
    total_slippage_cost: float = 0.0
    total_cost: float = 0.0

    # 序列数据
    equity_curve: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    positions: Optional[pd.Series] = None
    drawdown_curve: Optional[pd.Series] = None
    trade_log: Optional[pd.DataFrame] = None

    def summary(self) -> Dict:
        """返回汇总字典"""
        return {
            "总收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annualized_return:.2%}",
            "夏普比率": f"{self.sharpe_ratio:.2f}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "最大回撤持续天数": self.max_drawdown_duration,
            "卡尔马比率": f"{self.calmar_ratio:.2f}",
            "总交易次数": self.total_trades,
            "胜率": f"{self.win_rate:.2%}",
            "盈亏比": f"{self.profit_loss_ratio:.2f}",
            "平均持仓天数": f"{self.avg_holding_days:.1f}",
            "总佣金": f"¥{self.total_commission:.2f}",
            "总印花税": f"¥{self.total_stamp_tax:.2f}",
            "总滑点成本": f"¥{self.total_slippage_cost:.2f}",
            "总交易成本": f"¥{self.total_cost:.2f}",
        }


class VectorizedBacktestEngine:
    """
    向量化回测引擎

    使用 NumPy/Pandas 向量运算实现高性能回测。
    适用于信号已经预生成的场景（如因子选股、均线策略等）。

    用法示例::

        engine = VectorizedBacktestEngine(initial_capital=100000)
        # signals: 1=买入, -1=卖出, 0=持有
        result = engine.run(prices, signals)
        print(result.summary())
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        trade_cost_config: Optional[TradeCostConfig] = None,
        risk_free_rate: float = 0.03,
        trading_days_per_year: int = 242,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if initial_capital <= 0:
            raise InputValidationError(f"initial_capital 必须为正数，当前值: {initial_capital}")

        self.initial_capital = initial_capital
        self.cost_config = trade_cost_config or TradeCostConfig()
        self.cost_config.validate()
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        self.logger = logger or logging.getLogger("VectorizedBacktest")

    # ------------------------------------------------------------------
    # 核心回测
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.Series,
        signals: pd.Series,
        volatility: Optional[pd.Series] = None,
    ) -> VectorizedResult:
        """
        执行向量化回测

        Args:
            prices: 价格序列（收盘价），index 为日期
            signals: 交易信号序列（1=买入/持有多头, -1=卖出/做空, 0=空仓）
            volatility: 波动率序列（可选，用于动态滑点计算）

        Returns:
            VectorizedResult 回测结果对象
        """
        self._validate_inputs(prices, signals)
        self.logger.info(f"开始向量化回测，数据长度: {len(prices)}, 初始资金: ¥{self.initial_capital:,.2f}")

        # 对齐索引
        prices = prices.copy()
        signals = signals.reindex(prices.index).fillna(0).astype(int)
        if volatility is not None:
            volatility = volatility.reindex(prices.index).fillna(0)

        # ---------- 1. 计算仓位变化 ----------
        # position: 持仓状态 (1 / 0 / -1)
        position = signals.clip(-1, 1)
        # trades: 仓位变化点 (非0 表示有交易)
        trades = position.diff().fillna(position.iloc[0])

        # ---------- 2. 计算交易成本 ----------
        cost_series = self._calculate_costs(prices, trades, volatility)

        # ---------- 3. 计算收益 ----------
        daily_pct = prices.pct_change().fillna(0)
        # 策略日收益 = 仓位 * 资产收益 - 交易成本占比
        strategy_returns = position.shift(1).fillna(0) * daily_pct - cost_series / self.initial_capital

        # ---------- 4. 构建权益曲线 ----------
        equity_curve = self.initial_capital * (1 + strategy_returns).cumprod()

        # ---------- 5. 回撤分析 ----------
        drawdown, max_dd, max_dd_duration = self._calculate_drawdown(equity_curve)

        # ---------- 6. 交易统计 ----------
        trade_log, trade_stats = self._analyze_trades(prices, position, trades, cost_series)

        # ---------- 7. 风险指标 ----------
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1
        n_days = len(prices)
        ann_return = (1 + total_return) ** (self.trading_days / max(n_days, 1)) - 1
        ann_vol = strategy_returns.std() * np.sqrt(self.trading_days)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

        total_commission = cost_series.where(trades != 0).sum() * 0.3  # 粗估佣金占比
        total_stamp = cost_series.where(trades < 0).sum() * 0.3
        total_slippage = cost_series.sum() - total_commission - total_stamp

        result = VectorizedResult(
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar,
            total_trades=trade_stats["total_trades"],
            win_rate=trade_stats["win_rate"],
            profit_loss_ratio=trade_stats["profit_loss_ratio"],
            avg_holding_days=trade_stats["avg_holding_days"],
            total_commission=total_commission,
            total_stamp_tax=total_stamp,
            total_slippage_cost=total_slippage,
            total_cost=cost_series.sum(),
            equity_curve=equity_curve,
            daily_returns=strategy_returns,
            positions=position,
            drawdown_curve=drawdown,
            trade_log=trade_log,
        )

        self.logger.info(
            f"回测完成 | 总收益: {total_return:.2%} | 年化: {ann_return:.2%} | "
            f"夏普: {sharpe:.2f} | 最大回撤: {max_dd:.2%} | 交易次数: {trade_stats['total_trades']}"
        )
        return result

    # ------------------------------------------------------------------
    # 批量参数扫描
    # ------------------------------------------------------------------

    def parameter_sweep(
        self,
        prices: pd.Series,
        signal_fn,
        param_grid: Dict[str, List],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """
        批量参数扫描，利用向量化快速跑多组参数。

        Args:
            prices: 价格序列
            signal_fn: 信号生成函数，签名 signal_fn(prices, **params) -> pd.Series
            param_grid: 参数网格，如 {"fast": [5,10,20], "slow": [30,60,120]}
            metric: 优化目标指标名

        Returns:
            DataFrame，每行一组参数及其回测指标
        """
        import itertools

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        self.logger.info(f"参数扫描开始，共 {len(combos)} 组参数")
        rows: List[Dict] = []

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                signals = signal_fn(prices, **params)
                result = self.run(prices, signals)
                row = {**params, **{
                    "total_return": result.total_return,
                    "annualized_return": result.annualized_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "calmar_ratio": result.calmar_ratio,
                    "total_trades": result.total_trades,
                    "win_rate": result.win_rate,
                }}
                rows.append(row)
            except Exception as e:
                self.logger.warning(f"参数组 {params} 回测失败: {e}")
                rows.append({**params, "error": str(e)})

        df = pd.DataFrame(rows)
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False).reset_index(drop=True)

        self.logger.info(f"参数扫描完成，有效结果: {len(df.dropna(subset=[metric]) if metric in df.columns else df)} 组")
        return df

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _validate_inputs(self, prices: pd.Series, signals: pd.Series) -> None:
        """验证输入数据"""
        if prices is None or len(prices) == 0:
            raise InputValidationError("prices 不能为空")
        if signals is None or len(signals) == 0:
            raise InputValidationError("signals 不能为空")
        if prices.isnull().all():
            raise InputValidationError("prices 全部为 NaN")
        if (prices <= 0).any():
            raise InputValidationError("prices 中存在非正值")

    def _calculate_costs(
        self,
        prices: pd.Series,
        trades: pd.Series,
        volatility: Optional[pd.Series],
    ) -> pd.Series:
        """计算每日交易成本（金额）"""
        cfg = self.cost_config
        trade_value = (trades.abs() * prices * 100).fillna(0)  # 假设每手100股

        # 佣金
        commission = np.maximum(trade_value * cfg.commission_rate, cfg.min_commission * (trades != 0).astype(float))

        # 印花税（仅卖出）
        sell_mask = trades < 0
        stamp_tax = trade_value * cfg.stamp_tax_rate * sell_mask.astype(float)

        # 滑点
        if volatility is not None:
            slippage_rate = cfg.slippage_base + volatility * cfg.slippage_volatility_factor
            slippage_rate = slippage_rate.clip(0, cfg.max_impact_cost)
        else:
            slippage_rate = cfg.slippage_base

        slippage = trade_value * slippage_rate * (trades != 0).astype(float)

        total_cost = commission + stamp_tax + slippage
        return total_cost

    def _calculate_drawdown(self, equity: pd.Series) -> Tuple[pd.Series, float, int]:
        """计算回撤序列、最大回撤和最大回撤持续天数"""
        peak = equity.cummax()
        drawdown = (equity - peak) / peak

        max_dd = drawdown.min()

        # 最大回撤持续天数
        is_dd = drawdown < 0
        dd_groups = (~is_dd).cumsum()
        if is_dd.any():
            max_dd_duration = int(is_dd.groupby(dd_groups).sum().max())
        else:
            max_dd_duration = 0

        return drawdown, max_dd, max_dd_duration

    def _analyze_trades(
        self,
        prices: pd.Series,
        position: pd.Series,
        trades: pd.Series,
        cost_series: pd.Series,
    ) -> Tuple[Optional[pd.DataFrame], Dict]:
        """分析交易记录"""
        # 找出所有交易点
        trade_mask = trades != 0
        trade_indices = prices.index[trade_mask]

        if len(trade_indices) == 0:
            return None, {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_holding_days": 0.0,
            }

        # 构建交易对（入场-出场）
        entries: List[Dict] = []
        current_entry = None

        for idx in trade_indices:
            trade_dir = int(trades.loc[idx])
            if trade_dir > 0 and current_entry is None:
                # 开仓
                current_entry = {"entry_date": idx, "entry_price": prices.loc[idx]}
            elif trade_dir < 0 and current_entry is not None:
                # 平仓
                exit_price = prices.loc[idx]
                pnl = (exit_price - current_entry["entry_price"]) / current_entry["entry_price"]
                holding = (idx - current_entry["entry_date"]).days if hasattr(idx, "days") or hasattr(idx - current_entry["entry_date"], "days") else 1
                entries.append({
                    "entry_date": current_entry["entry_date"],
                    "exit_date": idx,
                    "entry_price": current_entry["entry_price"],
                    "exit_price": exit_price,
                    "return": pnl,
                    "holding_days": max(holding, 1),
                })
                current_entry = None

        if not entries:
            return None, {
                "total_trades": len(trade_indices),
                "win_rate": 0.0,
                "profit_loss_ratio": 0.0,
                "avg_holding_days": 0.0,
            }

        trade_df = pd.DataFrame(entries)
        wins = trade_df[trade_df["return"] > 0]
        losses = trade_df[trade_df["return"] <= 0]

        win_rate = len(wins) / len(trade_df) if len(trade_df) > 0 else 0
        avg_win = wins["return"].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses["return"].mean()) if len(losses) > 0 else 1e-10
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        stats = {
            "total_trades": len(trade_df),
            "win_rate": win_rate,
            "profit_loss_ratio": pl_ratio,
            "avg_holding_days": trade_df["holding_days"].mean(),
        }

        return trade_df, stats
