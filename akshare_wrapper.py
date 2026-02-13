#!/usr/bin/env python3
"""
AKShare 数据获取增强模块
增强稳定性、错误处理、数据校验和重试机制
符合量化工程标准的代码实现
"""

import akshare as ak
import pandas as pd
import numpy as np
import logging
import time
import json
import os
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from functools import wraps

# 日志配置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
logger.addHandler(console_handler)


@dataclass
class AkShareConfig:
    """AKShare配置参数"""
    max_retries: int = 10  # 最大重试次数
    retry_delay: float = 2.0  # 初始重试延迟(秒)
    timeout: int = 60  # 请求超时时间(秒)
    batch_size: int = 50  # 批量获取股票数量（减少批量大小以降低服务器压力）
    enable_cache: bool = True  # 是否启用缓存
    validate_data: bool = True  # 是否启用数据校验
    cache_dir: str = './data_cache'  # 缓存目录
    request_interval: float = 0.5  # 请求间隔(秒)，避免API频率限制（增加间隔）
    max_cache_age: int = 86400  # 缓存最大有效期(秒)，默认1天
    enable_network_check: bool = True  # 是否启用网络检查
    min_data_length: int = 10  # 最小数据长度要求
    max_concurrent_requests: int = 1  # 最大并发请求数


class AkShareWrapper:
    """AKShare数据获取增强包装类"""
    
    def __init__(self, config: Optional[AkShareConfig] = None):
        """
        初始化AKShare增强包装类
        
        Args:
            config: AKShare配置参数
        """
        self.config = config or AkShareConfig()
        self._setup_cache()
        self._last_request_time = 0
    
    def _setup_cache(self):
        """设置缓存目录"""
        if self.config.enable_cache:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def _throttle_requests(self):
        """请求节流控制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.request_interval:
            time.sleep(self.config.request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _check_network_availability(self) -> bool:
        """检查网络可用性"""
        try:
            import socket
            # 尝试连接到公共DNS服务器
            socket.create_connection(('8.8.8.8', 53), timeout=5)
            return True
        except Exception:
            return False
    
    @retry(
        stop=stop_after_attempt(10),  # 增加重试次数到10次
        wait=wait_exponential(multiplier=1, min=2, max=60),  # 增加重试间隔
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _safe_request(self, func, *args, **kwargs):
        """
        安全请求装饰器，添加重试机制
        
        Args:
            func: 要调用的AKShare函数
            args: 位置参数
            kwargs: 关键字参数
        
        Returns:
            函数返回值
        
        Raises:
            Exception: 重试失败后抛出异常
        """
        try:
            self._throttle_requests()
            logger.debug(f"调用API: {func.__name__}")
            result = func(*args, **kwargs)
            
            # 验证返回结果类型
            if result is None:
                raise ValueError("API返回None")
            
            return result
        except Exception as e:
            logger.warning(f"API调用失败: {e}, 正在重试...")
            # 对于某些特定的异常类型，增加额外的处理
            if "Connection aborted" in str(e) or "Remote end closed connection" in str(e):
                logger.warning("检测到连接中断，将增加重试间隔")
                time.sleep(5)
            raise
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_components = [func_name]
        for arg in args:
            key_components.append(str(arg))
        for k, v in sorted(kwargs.items()):
            key_components.append(f"{k}={v}")
        return "_".join(key_components).replace("/", "_").replace("\\", "_")
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.config.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存加载数据"""
        if not self.config.enable_cache:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查缓存是否过期
            timestamp = cache_data.get('timestamp', 0)
            if time.time() - timestamp > self.config.max_cache_age:
                logger.debug(f"缓存已过期: {cache_key}")
                try:
                    os.remove(cache_path)
                except:
                    pass
                return None
            
            logger.debug(f"从缓存加载数据: {cache_key}")
            df = pd.DataFrame(cache_data['data'])
            
            # 验证缓存数据的完整性
            if len(df) < 1:
                logger.debug(f"缓存数据为空: {cache_key}")
                try:
                    os.remove(cache_path)
                except:
                    pass
                return None
            
            return df
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            try:
                os.remove(cache_path)
            except:
                pass
            return None
    
    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        if not self.config.enable_cache:
            return
        
        try:
            cache_path = self._get_cache_path(cache_key)
            cache_data = {
                'timestamp': int(time.time()),
                'data': data.to_dict(orient='records')
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"数据已缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def _validate_dataframe(self, df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> bool:
        """
        增强的数据校验函数
        
        Args:
            df: 要校验的数据
            expected_columns: 期望的列名列表
        
        Returns:
            校验是否通过
        """
        if not self.config.validate_data:
            return True
        
        # 检查数据是否为空
        if df.empty:
            logger.warning("获取到空数据")
            return False
        
        # 检查数据类型是否为DataFrame
        if not isinstance(df, pd.DataFrame):
            logger.warning("返回结果不是DataFrame类型")
            return False
        
        # 检查关键列是否存在
        if expected_columns:
            missing_columns = set(expected_columns) - set(df.columns)
            if missing_columns:
                logger.warning(f"数据列缺失: {missing_columns}")
                return False
        
        # 检查数值列是否包含NaN或inf
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().any() or np.isinf(df[col]).any():
                logger.warning(f"数值列 {col} 包含NaN或inf值")
                # 尝试填充缺失值
                df[col] = df[col].fillna(df[col].mean())
        
        # 检查日期格式
        if '日期' in df.columns or 'date' in df.columns:
            date_column = '日期' if '日期' in df.columns else 'date'
            try:
                pd.to_datetime(df[date_column])
            except Exception as e:
                logger.warning(f"日期格式错误: {e}")
                return False
        
        # 检查数据完整性
        if len(df) < 10:
            logger.warning(f"数据量过少: 仅 {len(df)} 条记录")
        
        # 检查是否有重复索引
        if df.index.duplicated().any():
            logger.warning("数据存在重复索引")
            df = df[~df.index.duplicated(keep='first')]
        
        # 检查是否有重复行
        if df.duplicated().any():
            logger.warning("数据存在重复行")
            df = df.drop_duplicates()
        
        # 检查数据范围
        for col in numeric_columns:
            if df[col].dtype in ['int', 'float']:
                # 检查数值是否在合理范围内（防止数据异常）
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                if (df[col] < lower_bound).any() or (df[col] > upper_bound).any():
                    logger.warning(f"数值列 {col} 包含异常值（超出3倍IQR范围）")
        
        return True
    
    def get_stock_codes(self, universe: str = 'all') -> List[str]:
        """
        获取股票代码列表
        
        Args:
            universe: 股票池类型 ('hs300' | 'zz500' | 'all')
        
        Returns:
            股票代码列表
        """
        logger.info(f"正在获取股票池: {universe}")
        try:
            if universe == 'hs300':
                df = self._safe_request(ak.index_stock_cons, '000300')
                if not self._validate_dataframe(df, ['品种代码']):
                    return []
                return df['品种代码'].tolist()
            elif universe == 'zz500':
                df = self._safe_request(ak.index_stock_cons, '000905')
                if not self._validate_dataframe(df, ['品种代码']):
                    return []
                return df['品种代码'].tolist()
            elif universe == 'all':
                df = self._safe_request(ak.stock_info_a_code_name)
                if not self._validate_dataframe(df, ['code']):
                    return []
                return df['code'].tolist()
            else:
                raise ValueError(f"Unknown universe: {universe}")
        except Exception as e:
            logger.error(f"获取股票池失败: {e}")
            return []
    
    def get_stock_history(self, symbol: str, period: str = 'daily',
                          start_date: str = '20240101', end_date: str = '20250101') -> pd.DataFrame:
        """
        获取股票历史行情数据
        
        Args:
            symbol: 股票代码
            period: 周期 ('daily' | 'weekly' | 'monthly' | '15' | '60')
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            历史行情数据DataFrame
        """
        logger.info(f"正在获取 {symbol} 历史行情数据")
        
        # 检查缓存
        cache_key = self._get_cache_key('get_stock_history', symbol, period, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            df = self._safe_request(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date
            )
            
            if not self._validate_dataframe(df, ['日期', '开盘', '收盘', '最高', '最低', '成交量']):
                logger.warning(f"{symbol} 数据校验失败，返回空数据")
                return pd.DataFrame()
            
            # 数据清洗
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # 填充缺失值（使用前向填充）
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 移除无效的价格数据（例如负数）
            for col in ['开盘', '收盘', '最高', '最低']:
                if col in df.columns:
                    df = df[df[col] > 0]
            
            # 检查成交量是否为正数
            if '成交量' in df.columns:
                df = df[df['成交量'] > 0]
            
            # 检查日期范围是否符合要求
            df['日期'] = pd.to_datetime(df['日期'])
            mask = (df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))
            df = df[mask]
            
            # 保存到缓存
            self._save_to_cache(cache_key, df)
            
            logger.info(f"成功获取 {len(df)} 条数据")
            return df
        except Exception as e:
            logger.warning(f"获取 {symbol} 历史行情失败: {e}")
            # 尝试获取备用数据源或返回空数据
            return pd.DataFrame()
    
    def get_financial_indicator(self, symbol: str) -> pd.DataFrame:
        """
        获取股票财务分析指标
        
        Args:
            symbol: 股票代码
        
        Returns:
            财务指标数据DataFrame
        """
        logger.debug(f"正在获取 {symbol} 财务指标")
        
        # 检查缓存
        cache_key = self._get_cache_key('get_financial_indicator', symbol)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            df = self._safe_request(ak.stock_financial_analysis_indicator, symbol)
            
            if not self._validate_dataframe(df, ['日期', '市盈率-动态', '市净率-动态', '净资产收益率']):
                return pd.DataFrame()
            
            # 保存到缓存
            self._save_to_cache(cache_key, df)
            
            return df
        except Exception as e:
            logger.warning(f"获取 {symbol} 财务指标失败: {e}")
            return pd.DataFrame()
    
    def get_financial_data_batch(self, symbols: List[str], start_date: str,
                                 end_date: str, batch_size: int = 100) -> pd.DataFrame:
        """
        批量获取财务数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 批量大小
        
        Returns:
            合并后的财务数据DataFrame
        """
        logger.info(f"正在批量获取 {len(symbols)} 只股票的财务数据")
        
        all_data = []
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            logger.debug(f"处理批次 {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
            
            for symbol in batch_symbols:
                df = self.get_financial_indicator(symbol)
                if not df.empty:
                    # 时间过滤
                    df['date'] = pd.to_datetime(df['日期'])
                    mask = (df['date'] >= pd.to_datetime(start_date)) & \
                           (df['date'] <= pd.to_datetime(end_date))
                    filtered = df[mask]
                    
                    if len(filtered) > 0:
                        filtered['code'] = symbol
                        all_data.append(filtered)
                
                # 避免API频率限制
                time.sleep(0.1)
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"成功获取 {len(result['code'].unique())} 只股票的财务数据")
            return result
        
        logger.warning("未获取到任何财务数据")
        return pd.DataFrame()
    
    def get_index_components(self, index_code: str) -> pd.DataFrame:
        """
        获取指数成分股
        
        Args:
            index_code: 指数代码
        
        Returns:
            指数成分股DataFrame
        """
        logger.info(f"正在获取 {index_code} 成分股")
        
        # 检查缓存
        cache_key = self._get_cache_key('get_index_components', index_code)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            df = self._safe_request(ak.index_stock_cons, index_code)
            
            if not self._validate_dataframe(df, ['品种代码']):
                return pd.DataFrame()
            
            self._save_to_cache(cache_key, df)
            logger.info(f"成功获取 {len(df)} 只成分股")
            return df
        except Exception as e:
            logger.error(f"获取 {index_code} 成分股失败: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        获取股票基本信息
        
        Args:
            symbol: 股票代码
        
        Returns:
            股票基本信息字典
        """
        logger.debug(f"正在获取 {symbol} 基本信息")
        try:
            df = self._safe_request(ak.stock_info_a_code_name)
            stock_info = df[df['code'] == symbol]
            if not stock_info.empty:
                return stock_info.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"获取 {symbol} 基本信息失败: {e}")
        return None
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            交易日列表
        """
        logger.info("正在获取交易日历")
        
        # 检查缓存
        cache_key = self._get_cache_key('get_trading_dates', start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return list(cached_data['trade_date'])
        
        try:
            df = self._safe_request(ak.tool_trade_date_hist_sina)
            
            if not self._validate_dataframe(df, ['trade_date']):
                return []
            
            df['trade_date'] = df['trade_date'].astype(str)
            mask = (df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)
            trading_dates = df[mask]['trade_date'].tolist()
            
            # 保存到缓存
            self._save_to_cache(cache_key, df[mask])
            
            logger.info(f"成功获取 {len(trading_dates)} 个交易日")
            return trading_dates
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []
    
    def get_market_summary(self) -> Optional[Dict]:
        """
        获取市场概览数据
        
        Returns:
            市场概览数据字典
        """
        logger.info("正在获取市场概览")
        try:
            # 获取上证指数
            sh_data = self.get_stock_history('sh', 'daily', 
                                           (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                           datetime.now().strftime('%Y%m%d'))
            
            # 获取深证成指
            sz_data = self.get_stock_history('sz', 'daily',
                                           (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                           datetime.now().strftime('%Y%m%d'))
            
            # 获取创业板指
            cy_data = self.get_stock_history('399006', 'daily',
                                           (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                           datetime.now().strftime('%Y%m%d'))
            
            return {
                'sh': {
                    'name': '上证指数',
                    'latest': sh_data['收盘'].iloc[-1] if not sh_data.empty else 0,
                    'change': sh_data['涨跌幅'].iloc[-1] if not sh_data.empty else 0
                },
                'sz': {
                    'name': '深证成指',
                    'latest': sz_data['收盘'].iloc[-1] if not sz_data.empty else 0,
                    'change': sz_data['涨跌幅'].iloc[-1] if not sz_data.empty else 0
                },
                'cy': {
                    'name': '创业板指',
                    'latest': cy_data['收盘'].iloc[-1] if not cy_data.empty else 0,
                    'change': cy_data['涨跌幅'].iloc[-1] if not cy_data.empty else 0
                }
            }
        except Exception as e:
            logger.error(f"获取市场概览失败: {e}")
            return None


def test_akshare_enhanced():
    """测试AKShare增强模块"""
    logger.setLevel(logging.DEBUG)
    
    print("=" * 70)
    print("AKShare增强模块测试")
    print("=" * 70)
    
    # 创建配置
    config = AkShareConfig(
        max_retries=5,
        retry_delay=1,
        validate_data=True,
        enable_cache=True
    )
    
    # 初始化包装类
    ak_wrapper = AkShareWrapper(config)
    
    # 测试1: 获取股票池
    print("\n1. 测试股票池获取:")
    hs300_stocks = ak_wrapper.get_stock_codes('hs300')
    print(f"   沪深300成分股数量: {len(hs300_stocks)}")
    assert len(hs300_stocks) > 0, "沪深300成分股数量应为正数"
    
    # 测试2: 获取单只股票历史数据
    print("\n2. 测试单只股票历史数据获取:")
    if hs300_stocks:
        sample_stock = hs300_stocks[0]
        history_data = ak_wrapper.get_stock_history(sample_stock, 'daily', '20240101', '20240630')
        print(f"   {sample_stock} 历史数据条数: {len(history_data)}")
        assert len(history_data) > 0, "历史数据不应为空"
    
    # 测试3: 获取财务数据
    print("\n3. 测试财务数据获取:")
    if hs300_stocks:
        sample_stock = hs300_stocks[0]
        financial_data = ak_wrapper.get_financial_indicator(sample_stock)
        print(f"   {sample_stock} 财务数据条数: {len(financial_data)}")
    
    # 测试4: 批量获取财务数据
    print("\n4. 测试批量财务数据获取:")
    batch_data = ak_wrapper.get_financial_data_batch(hs300_stocks[:5], '20240101', '20240630')
    print(f"   批量获取财务数据条数: {len(batch_data)}")
    print(f"   覆盖股票数量: {len(batch_data['code'].unique())}")
    
    # 测试5: 获取交易日历
    print("\n5. 测试交易日历获取:")
    trading_dates = ak_wrapper.get_trading_dates('20240101', '20240630')
    print(f"   2024年上半年交易日数量: {len(trading_dates)}")
    assert len(trading_dates) > 0, "交易日数量应为正数"
    
    # 测试6: 获取指数成分股
    print("\n6. 测试指数成分股获取:")
    index_components = ak_wrapper.get_index_components('000300')
    print(f"   沪深300成分股数量: {len(index_components)}")
    assert len(index_components) > 0, "指数成分股数量应为正数"
    
    print("\n" + "=" * 70)
    print("✓ 所有测试通过!")
    print("=" * 70)


if __name__ == "__main__":
    test_akshare_enhanced()
