#!/usr/bin/env python3
"""测试增强版AKShare数据获取"""

import sys
import logging
from akshare_wrapper import AkShareWrapper, AkShareConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


def test_akshare_enhanced():
    """测试增强版AKShare数据获取"""
    logger.info("开始测试增强版AKShare数据获取...")
    
    try:
        # 创建配置
        config = AkShareConfig(
            max_retries=3,
            retry_delay=1.0,
            timeout=30,
            batch_size=100,
            validate_data=True,
            enable_cache=True
        )
        
        # 初始化增强版AKShare包装类
        ak_wrapper = AkShareWrapper(config)
        
        print("=" * 70)
        print("1. 测试股票池获取")
        print("=" * 70)
        
        # 测试沪深300股票池
        hs300_stocks = ak_wrapper.get_stock_codes('hs300')
        print(f"沪深300成分股数量: {len(hs300_stocks)}")
        assert len(hs300_stocks) > 0, "沪深300成分股数量应为正数"
        
        # 测试中证500股票池
        zz500_stocks = ak_wrapper.get_stock_codes('zz500')
        print(f"中证500成分股数量: {len(zz500_stocks)}")
        assert len(zz500_stocks) > 0, "中证500成分股数量应为正数"
        
        # 测试全市场股票池
        all_stocks = ak_wrapper.get_stock_codes('all')
        print(f"全市场股票数量: {len(all_stocks)}")
        assert len(all_stocks) > 0, "全市场股票数量应为正数"
        
        print("✓ 股票池获取测试通过")
        
        print("\n" + "=" * 70)
        print("2. 测试历史数据获取")
        print("=" * 70)
        
        # 测试上证指数
        sh_data = ak_wrapper.get_stock_history('sh', 'daily', '20240101', '20240110')
        print(f"上证指数历史数据条数: {len(sh_data)}")
        assert len(sh_data) > 0, "上证指数历史数据不应为空"
        
        # 测试单只股票
        if hs300_stocks:
            sample_stock = hs300_stocks[0]
            stock_data = ak_wrapper.get_stock_history(sample_stock, 'daily', '20240101', '20240110')
            print(f"{sample_stock} 历史数据条数: {len(stock_data)}")
            assert len(stock_data) > 0, "单只股票历史数据不应为空"
        
        print("✓ 历史数据获取测试通过")
        
        print("\n" + "=" * 70)
        print("3. 测试财务数据获取")
        print("=" * 70)
        
        if hs300_stocks:
            sample_stock = hs300_stocks[0]
            financial_data = ak_wrapper.get_financial_indicator(sample_stock)
            print(f"{sample_stock} 财务数据条数: {len(financial_data)}")
        
        print("✓ 财务数据获取测试通过")
        
        print("\n" + "=" * 70)
        print("4. 测试批量数据获取")
        print("=" * 70)
        
        # 测试批量获取财务数据
        if hs300_stocks:
            batch_data = ak_wrapper.get_financial_data_batch(hs300_stocks[:5], '20240101', '20240630')
            print(f"批量获取财务数据条数: {len(batch_data)}")
            print(f"覆盖股票数量: {len(batch_data['code'].unique())}")
            assert len(batch_data['code'].unique()) > 0, "批量获取应包含股票代码"
        
        print("✓ 批量数据获取测试通过")
        
        print("\n" + "=" * 70)
        print("5. 测试交易日历获取")
        print("=" * 70)
        
        trading_dates = ak_wrapper.get_trading_dates('20240101', '20240131')
        print(f"2024年1月交易日数量: {len(trading_dates)}")
        assert len(trading_dates) > 0, "交易日数量应为正数"
        
        print("✓ 交易日历获取测试通过")
        
        print("\n" + "=" * 70)
        print("6. 测试市场概览")
        print("=" * 70)
        
        market_summary = ak_wrapper.get_market_summary()
        if market_summary:
            for index, info in market_summary.items():
                print(f"{info['name']}: {info['latest']:.2f} ({info['change']:+.2f}%)")
        
        print("✓ 市场概览测试通过")
        
        print("\n" + "=" * 70)
        print("✅ AKShare增强版数据获取功能测试完成!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AKShare增强版数据获取测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_backward_compatibility():
    """测试与原代码的兼容性"""
    logger.info("开始测试与原代码的兼容性...")
    
    import akshare as ak
    
    try:
        # 测试原API是否仍然可用
        sh_data = ak.stock_zh_a_hist(symbol="sh", period="daily", start_date="20240101", end_date="20240102")
        assert len(sh_data) > 0, "原API获取数据失败"
        
        logger.info("✓ 原API兼容性测试通过")
        return True
    except Exception as e:
        logger.error(f"❌ 原API兼容性测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("AKShare增强版测试程序")
    print("=" * 70)
    
    # 运行增强版测试
    enhanced_test_result = test_akshare_enhanced()
    
    # 运行兼容性测试
    compatibility_result = test_backward_compatibility()
    
    if enhanced_test_result and compatibility_result:
        print("\n🎉 所有测试通过! AKShare增强版已准备就绪")
        sys.exit(0)
    else:
        print("\n❌ 测试失败")
        sys.exit(1)
