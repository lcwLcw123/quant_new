#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测引擎增强模块
功能：
- 动态滑点模拟
- 完整交易成本模型
- 前视偏差检测
- 凯利公式仓位管理
- 动态止损与浮动止盈
- 完善的错误处理和输入验证

Author: Coding Agent
Date: 2026-02-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from datetime import datetime
from pathlib import Path
import traceback


class BacktestError(Exception):
    """回测引擎自定义异常基类"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class InputValidationError(BacktestError):
    """输入参数验证错误"""
    pass


class TradeExecutionError(BacktestError):
    """交易执行错误"""
    pass


class PositionType(Enum):
    """持仓类型"""
    NONE = 0
    LONG = 1
    SHORT = -1


# 日志配置
def setup_logger(log_dir: Optional[Path] = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        log_dir: 日志目录（可选）
        log_level: 日志级别
        
    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger("EnhancedBacktestEngine")
    logger.setLevel(log_level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定目录）
    if log_dir:
        try:
            # 确保 log_dir 是 Path 对象
            log_dir = Path(log_dir) if isinstance(log_dir, str) else log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_handler = logging.FileHandler(log_dir / f"backtest_{timestamp}.log", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            logger.warning(f"无法创建日志文件处理器: {e}")
    
    return logger


@dataclass
class TradeCostConfig:
    """交易成本配置"""
    commission_rate: float = 0.0002      # 佣金费率 0.02%
    stamp_tax_rate: float = 0.001         # 印花税 0.1%（卖出时收取）
    min_commission: float = 5.0           # 最低佣金
    impact_cost_rate: float = 0.001       # 基础冲击成本 0.1%
    max_impact_cost: float = 0.005        # 最大冲击成本 0.5%
    slippage_base: float = 0.001          # 基础滑点 0.1%
    slippage_volatility_factor: float = 0.002  # 波动率滑点系数
    slippage_volume_factor: float = 0.001     # 成交量滑点系数
    
    def validate(self) -> None:
        """验证配置参数的有效性"""
        errors = []
        
        if not 0 <= self.commission_rate <= 1:
            errors.append(f"commission_rate必须在[0,1]范围内，当前值: {self.commission_rate}")
        if not 0 <= self.stamp_tax_rate <= 1:
            errors.append(f"stamp_tax_rate必须在[0,1]范围内，当前值: {self.stamp_tax_rate}")
        if self.min_commission < 0:
            errors.append(f"min_commission不能为负数，当前值: {self.min_commission}")
        if not 0 <= self.impact_cost_rate <= 1:
            errors.append(f"impact_cost_rate必须在[0,1]范围内，当前值: {self.impact_cost_rate}")
        if not 0 <= self.max_impact_cost <= 1:
            errors.append(f"max_impact_cost必须在[0,1]范围内，当前值: {self.max_impact_cost}")
        if not 0 <= self.slippage_base <= 1:
            errors.append(f"slippage_base必须在[0,1]范围内，当前值: {self.slippage_base}")
        
        if errors:
            raise InputValidationError(
                "TradeCostConfig参数验证失败",
                {"errors": errors, "config": {
                    'commission_rate': self.commission_rate,
                    'stamp_tax_rate': self.stamp_tax_rate,
                    'min_commission': self.min_commission,
                    'impact_cost_rate': self.impact_cost_rate,
                    'max_impact_cost': self.max_impact_cost,
                    'slippage_base': self.slippage_base
                }}
            )


@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: int
    trade_time: int
    trade_type: str  # 'BUY', 'SELL', 'CLOSE'
    price: float
    shares: int
    cost: float
    slippage: float
    commission: float
    stamp_tax: float
    impact_cost: float
    net_value: float
    accumulated_cost: float
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'trade_time': self.trade_time,
            'trade_type': self.trade_type,
            'price': round(self.price, 4),
            'shares': self.shares,
            'cost': round(self.cost, 2),
            'slippage': round(self.slippage, 4),
            'commission': round(self.commission, 2),
            'stamp_tax': round(self.stamp_tax, 2),
            'impact_cost': round(self.impact_cost, 2),
            'net_value': round(self.net_value, 2),
            'accumulated_cost': round(self.accumulated_cost, 2)
        }
    
    def validate(self) -> None:
        """验证交易记录的有效性"""
        if self.trade_id < 0:
            raise InputValidationError(f"trade_id不能为负数: {self.trade_id}")
        if self.trade_time < 0:
            raise InputValidationError(f"trade_time不能为负数: {self.trade_time}")
        if self.trade_type not in ('BUY', 'SELL', 'CLOSE'):
            raise InputValidationError(f"无效的交易类型: {self.trade_type}")
        if self.price < 0:
            raise InputValidationError(f"price不能为负数: {self.price}")
        if self.shares <= 0:
            raise InputValidationError(f"shares必须为正数: {self.shares}")
        if self.commission < 0:
            raise InputValidationError(f"commission不能为负数: {self.commission}")
        if self.stamp_tax < 0:
            raise InputValidationError(f"stamp_tax不能为负数: {self.stamp_tax}")


@dataclass
class Position:
    """持仓信息"""
    shares: int = 0
    avg_price: float = 0.0
    position_type: PositionType = PositionType.NONE
    entry_time: int = 0
    entry_price: float = 0.0
    stop_loss: float = 0.0  # 止损价
    take_profit: float = 0.0  # 止盈价
    trailing_stop: float = 0.0  # 移动止损价
    peak_price: float = 0.0  # 持仓期间最高价/最低价
    
    @property
    def market_value(self) -> float:
        """市值"""
        return self.shares * self.avg_price
    
    def validate(self) -> None:
        """验证持仓信息的有效性"""
        if self.shares < 0:
            raise InputValidationError(f"shares不能为负数: {self.shares}")
        if self.avg_price < 0:
            raise InputValidationError(f"avg_price不能为负数: {self.avg_price}")
        if self.entry_price < 0:
            raise InputValidationError(f"entry_price不能为负数: {self.entry_price}")


class EnhancedBacktestEngine:
    """
    增强型回测引擎
    
    特性：
    - 动态滑点计算
    - 完整交易成本建模
    - 前视偏差检测
    - 凯利公式仓位管理
    - 动态止损/浮动止盈
    - 结构化日志记录
    - 完善的错误处理
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        trade_cost_config: Optional[TradeCostConfig] = None,
        kelly_fraction: float = 0.5,
        risk_free_rate: float = 0.03,
        verbose: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            trade_cost_config: 交易成本配置
            kelly_fraction: 凯利系数（降低风险用）
            risk_free_rate: 年化无风险利率
            verbose: 是否输出详细信息
            logger: 日志记录器实例（可选）
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trade_cost_config = trade_cost_config or TradeCostConfig()
        self.kelly_fraction = kelly_fraction
        self.risk_free_rate = risk_free_rate
        self.verbose = verbose
        
        # 日志记录器
        self.logger = logger or logging.getLogger("EnhancedBacktestEngine")
        
        # 状态变量
        self.position = Position()
        self.trades: List[TradeRecord] = []
        self.trade_id_counter = 0
        self.look_ahead_bias_detected = False
        self.look_ahead_bias_warnings: List[str] = []
        
        # 评估指标缓存
        self._equity_curve: List[float] = []
        self._daily_returns: List[float] = []
        self._position_values: List[float] = []
        
        self.logger.info(f"回测引擎初始化完成，初始资金: ¥{initial_capital:,.2f}")
        
    def _log(self, *args, **kwargs) -> None:
        """日志输出（兼容旧接口）"""
        if self.verbose:
            message = " ".join(str(arg) for arg in args)
            self.logger.info(message)
    
    def calculate_slippage(
        self,
        price: float,
        volatility: float,
        volume_ratio: float,
        position_size: float,
        total_market_cap: float = 1e9
    ) -> float:
        """
        计算动态滑点
        
        Args:
            price: 交易价格
            volatility: 波动率（标准差）
            volume_ratio: 成交量比率（当前/平均）
            position_size: 仓位大小
            total_market_cap: 总市值
            
        Returns:
            滑点比例（0-1）
        """
        config = self.trade_cost_config
        
        # 基于波动率的滑点
        vol_slippage = min(
            volatility * config.slippage_volatility_factor * 100,
            config.max_impact_cost
        )
        
        # 基于成交量的滑点
        volume_slippage = min(
            (position_size / total_market_cap) * volume_ratio * 100 * config.slippage_volume_factor,
            config.max_impact_cost
        )
        
        # 基于仓位的滑点（冲击成本）
        size_slippage = min(
            (position_size / total_market_cap) * 10 * config.impact_cost_rate,
            config.max_impact_cost
        )
        
        # 基础滑点
        base_slippage = config.slippage_base
        
        # 总滑点
        total_slippage = base_slippage + vol_slippage + volume_slippage + size_slippage
        total_slippage = min(total_slippage, 0.05)  # 最高5%滑点
        
        return total_slippage
    
    def calculate_commission(self, trade_value: float) -> float:
        """
        计算佣金
        
        Args:
            trade_value: 交易金额
            
        Returns:
            佣金金额
        """
        commission = trade_value * self.trade_cost_config.commission_rate
        return max(commission, self.trade_cost_config.min_commission)
    
    def calculate_stamp_tax(self, trade_value: float) -> float:
        """
        计算印花税（卖出时收取）
        
        Args:
            trade_value: 交易金额
            
        Returns:
            印花税金额
        """
        return trade_value * self.trade_cost_config.stamp_tax_rate
    
    def calculate_impact_cost(
        self,
        price: float,
        shares: int,
        avg_daily_volume: float,
        volatility: float
    ) -> float:
        """
        计算冲击成本
        
        Args:
            price: 价格
            shares: 成交量
            avg_daily_volume: 日均成交量
            volatility: 波动率
            
        Returns:
            冲击成本金额
        """
        if avg_daily_volume <= 0:
            return 0
        
        config = self.trade_cost_config
        
        # 成交量占比
        volume_ratio = shares / (avg_daily_volume * 100)  # 假设avg_daily_volume是手数
        
        # 冲击成本与成交量占比成正比，与流动性成反比
        impact_rate = min(
            volume_ratio * config.impact_cost_rate * (1 + volatility * 10),
            config.max_impact_cost
        )
        
        return price * shares * impact_rate
    
    def calculate_kelly_position(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_position_pct: float = 0.25
    ) -> float:
        """
        使用凯利公式计算最优仓位比例
        
        凯利公式: f* = (p * b - q) / b
        其中：p=胜率, q=败率(1-p), b=盈亏比
        
        Args:
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损（正数）
            max_position_pct: 最大仓位比例限制
            
        Returns:
            建议仓位比例（0-1）
        """
        if avg_loss == 0:
            return max_position_pct
        
        # 盈亏比
        win_loss_ratio = avg_win / avg_loss
        
        # 凯利公式
        kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # 调整系数降低风险
        kelly_pct *= self.kelly_fraction
        
        # 限制在合理范围
        kelly_pct = max(0, min(kelly_pct, max_position_pct))
        
        return kelly_pct
    
    def detect_look_ahead_bias(self, df: pd.DataFrame) -> List[str]:
        """
        检测前视偏差
        
        Args:
            df: 包含未来数据的DataFrame
            
        Returns:
            警告列表
        """
        warnings = []
        
        # 检查是否使用未来函数
        future_indicators = [
            'future_return',
            'future_high',
            'future_low',
            'shift(-1)',  # 使用了shift(-1)获取未来数据
            'rolling(0)',  # 错误使用
        ]
        
        # 检查DataFrame列名
        for col in df.columns:
            if any(future_ind in col.lower() for future_ind in future_indicators):
                warnings.append(f"潜在前视偏差：列 '{col}' 可能包含未来信息")
        
        # 检测策略信号是否使用了未来数据
        if 'predicted' in df.columns and 'next_return' in df.columns:
            # 这是正常的特征工程，但需要确保预测是在回测前完成的
            warnings.append("检测到 'next_return' 列，请确保预测信号基于历史数据生成")
        
        self.look_ahead_bias_warnings = warnings
        self.look_ahead_bias_detected = len(warnings) > 0
        
        return warnings
    
    def check_stop_loss(self, current_price: float) -> bool:
        """
        检查是否触发止损
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否触发止损
        """
        if self.position.position_type == PositionType.LONG:
            return current_price <= self.position.stop_loss
        elif self.position.position_type == PositionType.SHORT:
            return current_price >= self.position.stop_loss
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """
        检查是否触发止盈
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否触发止盈
        """
        if self.position.position_type == PositionType.LONG:
            return current_price >= self.position.take_profit
        elif self.position.position_type == PositionType.SHORT:
            return current_price <= self.position.take_profit
        return False
    
    def update_trailing_stop(self, current_price: float) -> None:
        """
        更新移动止损线
        
        Args:
            current_price: 当前价格
        """
        if self.position.position_type == PositionType.LONG:
            # 多头：更新最高价，跟随止盈
            self.position.peak_price = max(self.position.peak_price, current_price)
            self.position.trailing_stop = self.position.peak_price * (1 - self.trailing_stop_pct)
        elif self.position.position_type == PositionType.SHORT:
            # 空头：更新最低价
            self.position.peak_price = min(self.position.peak_price, current_price)
            self.position.trailing_stop = self.position.peak_price * (1 + self.trailing_stop_pct)
    
    def check_trailing_stop(self, current_price: float) -> bool:
        """
        检查是否触发移动止损
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否触发移动止损
        """
        if self.position.position_type == PositionType.LONG:
            return current_price <= self.position.trailing_stop
        elif self.position.position_type == PositionType.SHORT:
            return current_price >= self.position.trailing_stop
        return False
    
    def execute_trade(
        self,
        trade_time: int,
        trade_type: str,
        price: float,
        shares: int,
        volatility: float = 0.02,
        volume_ratio: float = 1.0,
        avg_daily_volume: float = 1e7,
        accumulated_cost: float = 0.0
    ) -> Optional[TradeRecord]:
        """
        执行交易（包含完整成本计算）
        
        Args:
            trade_time: 交易时间索引
            trade_type: 'BUY' or 'SELL'
            price: 交易价格
            shares: 股份数量
            volatility: 波动率
            volume_ratio: 成交量比率
            avg_daily_volume: 日均成交量
            accumulated_cost: 累计成本
            
        Returns:
            交易记录
        """
        # 计算滑点
        position_size = price * shares
        slippage_rate = self.calculate_slippage(price, volatility, volume_ratio, position_size)
        
        # 根据买卖方向应用滑点
        if trade_type == 'BUY':
            # 买入时：实际买入价 = 基准价 * (1 + 滑点)
            actual_price = price * (1 + slippage_rate)
        else:  # SELL
            # 卖出时：实际卖出价 = 基准价 * (1 - 滑点)
            actual_price = price * (1 - slippage_rate)
        
        # 计算各项成本
        trade_value = actual_price * shares
        commission = self.calculate_commission(trade_value)
        stamp_tax = self.calculate_stamp_tax(trade_value) if trade_type == 'SELL' else 0
        impact_cost = self.calculate_impact_cost(price, shares, avg_daily_volume, volatility)
        
        # 总成本
        total_cost = commission + stamp_tax + impact_cost + abs(trade_value - position_size)
        
        # 更新累计成本
        accumulated_cost += total_cost
        
        # 创建交易记录
        trade = TradeRecord(
            trade_id=self.trade_id_counter,
            trade_time=trade_time,
            trade_type=trade_type,
            price=price,
            shares=shares,
            cost=trade_value + total_cost if trade_type == 'BUY' else trade_value - total_cost,
            slippage=abs(trade_value - position_size),
            commission=commission,
            stamp_tax=stamp_tax,
            impact_cost=impact_cost,
            net_value=trade_value,
            accumulated_cost=accumulated_cost
        )
        
        self.trade_id_counter += 1
        self.trades.append(trade)
        
        return trade
    
    def run_backtest(
        self,
        prices: np.ndarray,
        signals: np.ndarray,
        volatility: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        stop_loss_pct: float = 0.05,
        trailing_stop_pct: float = 0.03,
        trailing_stop_activation: float = 0.05,
        use_kelly: bool = True,
        use_trailing_stop: bool = True,
        risk_per_trade: float = 0.02,
        trade_size_multiplier: float = 1.0
    ) -> Dict:
        """
        运行回测
        
        Args:
            prices: 价格序列
            signals: 信号序列（1=买入，-1=卖出，0=持有）
            volatility: 波动率序列
            volumes: 成交量序列
            stop_loss_pct: 固定止损比例
            trailing_stop_pct: 移动止损比例
            trailing_stop_activation: 激活移动止损的盈利比例
            use_kelly: 是否使用凯利公式
            use_trailing_stop: 是否使用移动止损
            risk_per_trade: 每笔交易风险比例
            trade_size_multiplier: 仓位大小乘数
            
        Returns:
            回测结果字典
        """
        n = len(prices)
        
        # 默认波动率和成交量
        if volatility is None:
            volatility = np.full(n, 0.02)
        if volumes is None:
            volumes = np.full(n, 1e7)
        
        avg_daily_volume = np.mean(volumes)
        
        # 初始化
        self.capital = self.initial_capital
        self.position = Position()
        self.trades = []
        self._equity_curve = [self.capital]
        self._position_values = []
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_stop_activation = trailing_stop_activation
        
        # 计算交易成本统计
        total_commission = 0
        total_stamp_tax = 0
        total_impact_cost = 0
        total_slippage = 0
        
        # 主循环
        for i in range(n):
            current_price = prices[i]
            signal = signals[i] if i < len(signals) else 0
            current_vol = volatility[i]
            current_vol_ratio = volumes[i] / (avg_daily_volume + 1e-8)
            
            # 更新持仓状态
            if self.position.position_type != PositionType.NONE:
                self._position_values.append(self.position.market_value)
                
                # 更新移动止损
                if use_trailing_stop:
                    self.update_trailing_stop(current_price)
                
                # 检查止损/止盈条件
                if self.check_stop_loss(current_price):
                    # 止损平仓
                    trade = self.execute_trade(
                        i, 'SELL', current_price, self.position.shares,
                        current_vol, current_vol_ratio, avg_daily_volume,
                        total_commission + total_stamp_tax + total_impact_cost
                    )
                    if trade:
                        total_commission += trade.commission
                        total_stamp_tax += trade.stamp_tax
                        total_impact_cost += trade.impact_cost
                        total_slippage += trade.slippage
                        
                        self.capital += self.position.shares * current_price - trade.commission - trade.stamp_tax - trade.impact_cost - trade.slippage
                        
                        self._log(f"[{i}] 止损平仓: ¥{current_price:.2f}")
                        
                        self.position = Position()
                        
                elif self.check_take_profit(current_price):
                    # 止盈平仓
                    trade = self.execute_trade(
                        i, 'SELL', current_price, self.position.shares,
                        current_vol, current_vol_ratio, avg_daily_volume,
                        total_commission + total_stamp_tax + total_impact_cost
                    )
                    if trade:
                        total_commission += trade.commission
                        total_stamp_tax += trade.stamp_tax
                        total_impact_cost += trade.impact_cost
                        total_slippage += trade.slippage
                        
                        self.capital += self.position.shares * current_price - trade.commission - trade.stamp_tax - trade.impact_cost - trade.slippage
                        
                        self._log(f"[{i}] 止盈平仓: ¥{current_price:.2f}")
                        
                        self.position = Position()
                        
                elif use_trailing_stop and self.check_trailing_stop(current_price):
                    # 移动止损平仓
                    trade = self.execute_trade(
                        i, 'SELL', current_price, self.position.shares,
                        current_vol, current_vol_ratio, avg_daily_volume,
                        total_commission + total_stamp_tax + total_impact_cost
                    )
                    if trade:
                        total_commission += trade.commission
                        total_stamp_tax += trade.stamp_tax
                        total_impact_cost += trade.impact_cost
                        total_slippage += trade.slippage
                        
                        self.capital += self.position.shares * current_price - trade.commission - trade.stamp_tax - trade.impact_cost - trade.slippage
                        
                        self._log(f"[{i}] 移动止损平仓: ¥{current_price:.2f}")
                        
                        self.position = Position()
            else:
                self._position_values.append(0)
            
            # 买入信号处理
            if signal > 0 and self.position.position_type == PositionType.NONE:
                # 计算仓位大小
                if use_kelly:
                    # 使用凯利公式（这里用简化版本，实际应基于历史统计）
                    kelly_pct = self.calculate_kelly_position(
                        win_rate=0.55,  # 假设胜率
                        avg_win=0.03,    # 假设平均盈利3%
                        avg_loss=0.02    # 假设平均亏损2%
                    )
                    position_pct = min(kelly_pct * trade_size_multiplier, 0.5)
                else:
                    position_pct = risk_per_trade * trade_size_multiplier
                
                # 根据止损比例调整仓位
                adjusted_position_pct = position_pct
                if stop_loss_pct > 0:
                    # 仓位 = 风险金额 / (入场价 * 止损比例)
                    risk_amount = self.capital * risk_per_trade
                    position_value = risk_amount / stop_loss_pct
                    adjusted_position_pct = min(position_value / self.capital, 0.5)
                
                # 计算买入股数（100股整数倍）
                available_capital = self.capital * adjusted_position_pct
                shares = int(available_capital / (current_price * 100)) * 100
                shares = max(shares, 100)  # 至少100股
                
                if shares > 0:
                    trade = self.execute_trade(
                        i, 'BUY', current_price, shares,
                        current_vol, current_vol_ratio, avg_daily_volume,
                        total_commission + total_stamp_tax + total_impact_cost
                    )
                    if trade:
                        total_commission += trade.commission
                        total_impact_cost += trade.impact_cost
                        total_slippage += trade.slippage
                        
                        self.position = Position(
                            shares=shares,
                            avg_price=current_price,
                            position_type=PositionType.LONG,
                            entry_time=i,
                            entry_price=current_price,
                            stop_loss=current_price * (1 - stop_loss_pct),
                            take_profit=current_price * (1 + stop_loss_pct * 2),
                            peak_price=current_price,
                            trailing_stop=current_price * (1 - trailing_stop_pct)
                        )
                        
                        self.capital -= (shares * current_price + trade.commission + trade.impact_cost + trade.slippage)
                        
                        self._log(f"[{i}] 买入建仓: ¥{current_price:.2f}, 股数: {shares}, 止损: ¥{self.position.stop_loss:.2f}")
            
            # 更新权益曲线
            position_value = self.position.market_value if self.position.position_type != PositionType.NONE else 0
            self._equity_curve.append(self.capital + position_value)
        
        # 最终平仓
        if self.position.position_type != PositionType.NONE:
            final_price = prices[-1]
            trade = self.execute_trade(
                n-1, 'CLOSE', final_price, self.position.shares,
                volatility[-1], volumes[-1]/avg_daily_volume, avg_daily_volume,
                total_commission + total_stamp_tax + total_impact_cost
            )
            if trade:
                total_commission += trade.commission
                total_stamp_tax += trade.stamp_tax
                total_impact_cost += trade.impact_cost
                total_slippage += trade.slippage
            
            self.capital += self.position.shares * final_price - trade.commission - trade.stamp_tax - trade.impact_cost - trade.slippage
            self.position = Position()
        
        # 计算评估指标
        result = self._calculate_metrics(
            total_commission, total_stamp_tax, total_impact_cost, total_slippage
        )
        
        return result
    
    def _calculate_metrics(
        self,
        total_commission: float,
        total_stamp_tax: float,
        total_impact_cost: float,
        total_slippage: float
    ) -> Dict:
        """
        计算回测评估指标
        
        Returns:
            包含所有指标的字典
        """
        equity_curve = np.array(self._equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        self._daily_returns = returns
        
        # 基本指标
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        n_periods = len(equity_curve)
        annual_return = (1 + total_return) ** (252 * 78 / n_periods) - 1  # 假设每天78个15分钟K线
        
        # 年化波动率
        annual_volatility = np.std(returns) * np.sqrt(252 * 78)
        
        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / (annual_volatility + 1e-8)
        
        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 交易统计
        buy_trades = [t for t in self.trades if t.trade_type in ('BUY',)]
        sell_trades = [t for t in self.trades if t.trade_type in ('SELL', 'CLOSE')]
        completed_trades = []
        
        for i, buy_trade in enumerate(buy_trades):
            for j, sell_trade in enumerate(sell_trades):
                if j > i and sell_trade.trade_time > buy_trade.trade_time:
                    profit = (sell_trade.price - buy_trade.price) * buy_trade.shares
                    profit -= (buy_trade.commission + sell_trade.commission + 
                              sell_trade.stamp_tax + buy_trade.impact_cost + sell_trade.impact_cost +
                              buy_trade.slippage + sell_trade.slippage)
                    completed_trades.append({
                        'entry': buy_trade.trade_time,
                        'exit': sell_trade.trade_time,
                        'entry_price': buy_trade.price,
                        'exit_price': sell_trade.price,
                        'profit': profit,
                        'return': profit / (buy_trade.price * buy_trade.shares)
                    })
                    break
        
        # 胜率
        if completed_trades:
            wins = sum(1 for t in completed_trades if t['profit'] > 0)
            win_rate = wins / len(completed_trades)
            avg_win = np.mean([t['profit'] for t in completed_trades if t['profit'] > 0]) or 0
            avg_loss = np.mean([t['profit'] for t in completed_trades if t['profit'] < 0]) or 0
            profit_factor = abs(avg_win * wins / (avg_loss * (len(completed_trades) - wins) + 1e-8))
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # 卡尔马比率（年化收益/最大回撤）
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 交易成本占比
        total_costs = total_commission + total_stamp_tax + total_impact_cost + total_slippage
        cost_ratio = total_costs / self.initial_capital if self.initial_capital > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'total_trades': len(completed_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_commission': total_commission,
            'total_stamp_tax': total_stamp_tax,
            'total_impact_cost': total_impact_cost,
            'total_slippage': total_slippage,
            'total_costs': total_costs,
            'cost_ratio': cost_ratio,
            'equity_curve': equity_curve.tolist(),
            'trades': [t.to_dict() for t in self.trades],
            'completed_trades': completed_trades,
            'look_ahead_bias_warnings': self.look_ahead_bias_warnings,
            'look_ahead_bias_detected': self.look_ahead_bias_detected
        }
    
    def print_report(self, result: Dict, title: str = "回测报告") -> None:
        """
        打印回测报告
        
        Args:
            result: 回测结果字典
            title: 报告标题
        """
        print("\n" + "="*70)
        print(f"                 {title}")
        print("="*70)
        print(f"""
【收益概览】
  初始资金:           ¥{result['initial_capital']:>15,.2f}
  最终资金:           ¥{result['final_capital']:>15,.2f}
  总收益率:           {result['total_return']*100:>14.2f}%
  年化收益率:         {result['annual_return']*100:>14.2f}%

【风险指标】
  年化波动率:         {result['annual_volatility']*100:>14.2f}%
  最大回撤:           {result['max_drawdown']*100:>14.2f}%
  夏普比率:           {result['sharpe_ratio']:>15.4f}
  卡尔马比率:         {result['calmar_ratio']:>15.4f}

【交易统计】
  总交易次数:         {result['total_trades']:>15d}
  胜率:               {result['win_rate']*100:>14.2f}%
  平均盈利:           ¥{result['avg_win']:>14,.2f}
  平均亏损:           ¥{result['avg_loss']:>14,.2f}
  盈亏比:             {result['profit_factor']:>15.4f}

【交易成本】
  佣金合计:           ¥{result['total_commission']:>14,.2f}
  印花税合计:         ¥{result['total_stamp_tax']:>14,.2f}
  冲击成本合计:       ¥{result['total_impact_cost']:>14,.2f}
  滑点合计:           ¥{result['total_slippage']:>14,.2f}
  总成本:             ¥{result['total_costs']:>14,.2f}
  成本占比:           {result['cost_ratio']*100:>14.2f}%
""")
        
        if result['look_ahead_bias_detected']:
            print("【⚠️ 前视偏差警告】")
            for warning in result['look_ahead_bias_warnings']:
                print(f"  - {warning}")
        else:
            print("【✓ 前视偏差检查】: 未检测到前视偏差")
        
        print("="*70)


def create_enhanced_backtest(
    prices: np.ndarray,
    signals: np.ndarray,
    initial_capital: float = 100000,
    use_kelly: bool = True,
    stop_loss_pct: float = 0.05,
    trailing_stop_pct: float = 0.03,
    trailing_stop_activation: float = 0.05,
    **kwargs
) -> Tuple[Dict, 'EnhancedBacktestEngine']:
    """
    便捷函数：创建并运行增强回测
    
    Args:
        prices: 价格序列
        signals: 信号序列
        initial_capital: 初始资金
        use_kelly: 是否使用凯利公式
        stop_loss_pct: 止损比例
        trailing_stop_pct: 移动止损比例
        **kwargs: 其他参数
        
    Returns:
        (结果字典, 回测引擎实例)
    """
    engine = EnhancedBacktestEngine(
        initial_capital=initial_capital,
        verbose=kwargs.get('verbose', True)
    )
    
    result = engine.run_backtest(
        prices=prices,
        signals=signals,
        volatility=volatility,
        volumes=volumes,
        use_kelly=use_kelly,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        trailing_stop_activation=trailing_stop_activation
    )
    
    return result, engine


# ========== 演示代码 ==========
if __name__ == "__main__":
    # 模拟数据测试
    np.random.seed(42)
    
    # 生成模拟价格数据
    n = 500
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.cumprod(1 + returns)
    prices = np.maximum(prices, 10)  # 确保价格不为负
    
    # 模拟波动率和成交量
    volatility = np.abs(returns) * 2 + 0.01
    volumes = np.random.uniform(1e7, 5e7, n)
    
    # 模拟交易信号（基于简单均线交叉）
    ma_short = pd.Series(prices).rolling(10).mean().values
    ma_long = pd.Series(prices).rolling(30).mean().values
    
    signals = np.zeros(n)
    for i in range(1, n):
        if ma_short[i] > ma_long[i] and ma_short[i-1] <= ma_long[i-1]:
            signals[i] = 1  # 买入信号
        elif ma_short[i] < ma_long[i] and ma_short[i-1] >= ma_long[i-1]:
            signals[i] = -1  # 卖出信号
    
    print("="*70)
    print("          增强回测引擎 - 演示测试")
    print("="*70)
    
    # 运行回测
    result, engine = create_enhanced_backtest(
        prices=prices,
        signals=signals,
        initial_capital=100000,
        volatility=volatility,
        volumes=volumes,
        use_kelly=True,
        stop_loss_pct=0.05,
        trailing_stop_pct=0.03,
        trailing_stop_activation=0.05,
        verbose=True
    )
    
    # 打印报告
    engine.print_report(result, "增强回测引擎 - 演示报告")
    
    # 前视偏差检测演示
    print("\n【前视偏差检测演示】")
    df_demo = pd.DataFrame({
        'price': prices[:50],
        'predicted': np.random.rand(50),
        'future_return': np.random.rand(50)  # 模拟未来收益列
    })
    warnings = engine.detect_look_ahead_bias(df_demo)
    for w in warnings:
        print(f"  ⚠️ {w}")
    
    print("\n✓ 演示完成!")
