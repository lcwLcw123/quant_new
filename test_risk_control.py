
#!/usr/bin/env python3
"""
风险控制模块测试脚本
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('/Users/hanyinghui/quant_new')

# 导入风控相关函数
from quant_system import get_industry_classification, calculate_style_factors, neutralize_factors, control_style_exposure

def test_industry_classification():
    """测试行业分类功能"""
    print("测试行业分类功能...")
    try:
        # 创建模拟数据
        np.random.seed(42)
        dates = [datetime(2024, 1, i) for i in range(1, 11)]
        stock_codes = [f"60000{i}" for i in range(1, 11)]
        data = []
        for date in dates:
            for code in stock_codes:
                data.append({
                    '日期': date,
                    '股票代码': code,
                    '开盘': np.random.uniform(10, 100),
                    '最高': np.random.uniform(10, 100),
                    '最低': np.random.uniform(10, 100),
                    '收盘': np.random.uniform(10, 100),
                    '成交量': np.random.uniform(1e6, 1e8),
                    '成交额': np.random.uniform(1e7, 1e9),
                    '振幅': np.random.uniform(0, 10),
                    '涨跌幅': np.random.uniform(-5, 5),
                    '涨跌额': np.random.uniform(-5, 5),
                    '换手率': np.random.uniform(0, 20)
                })
        
        df = pd.DataFrame(data)
        
        # 测试行业分类
        industry_df = get_industry_classification(stock_codes)
        print(f"✓ 成功获取行业分类，包含 {len(industry_df)} 只股票")
        print("  前5个行业:")
        for i, (_, row) in enumerate(industry_df.head().iterrows()):
            print(f"  {i+1}. 股票: {row['code']}, 行业: {row['industry']}")
        
        # 验证行业分类覆盖
        assert len(industry_df['code'].unique()) == len(df['股票代码'].unique())
        print("✓ 行业分类覆盖所有股票")
        
        return True
    except Exception as e:
        print(f"✗ 行业分类功能测试失败: {e}")
        return False

def test_style_factors():
    """测试风格因子计算功能"""
    print("\n测试风格因子计算功能...")
    try:
        # 创建模拟数据
        np.random.seed(42)
        dates = [datetime(2024, 1, i) for i in range(1, 11)]
        stock_codes = [f"60000{i}" for i in range(1, 11)]
        data = []
        for date in dates:
            for code in stock_codes:
                data.append({
                    '日期': date,
                    '股票代码': code,
                    '开盘': np.random.uniform(10, 100),
                    '最高': np.random.uniform(10, 100),
                    '最低': np.random.uniform(10, 100),
                    '收盘': np.random.uniform(10, 100),
                    '成交量': np.random.uniform(1e6, 1e8),
                    '成交额': np.random.uniform(1e7, 1e9),
                    '振幅': np.random.uniform(0, 10),
                    '涨跌幅': np.random.uniform(-5, 5),
                    '涨跌额': np.random.uniform(-5, 5),
                    '换手率': np.random.uniform(0, 20)
                })
        
        df = pd.DataFrame(data)
        
        # 测试风格因子计算
        df_style = calculate_style_factors(df)
        style_factors = ['market_cap', 'valuation', 'growth', 'quality', 'momentum']
        
        print(f"✓ 成功计算风格因子，包含 {len(style_factors)} 个风格因子")
        for factor in style_factors:
            assert factor in df_style.columns
            print(f"  {factor}: 均值={df_style[factor].mean():.2f}, 标准差={df_style[factor].std():.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 风格因子计算功能测试失败: {e}")
        return False

def test_factor_neutralization():
    """测试因子中性化功能"""
    print("\n测试因子中性化功能...")
    try:
        # 创建模拟数据
        np.random.seed(42)
        dates = [datetime(2024, 1, i) for i in range(1, 11)]
        stock_codes = [f"60000{i}" for i in range(1, 11)]
        data = []
        for date in dates:
            for code in stock_codes:
                data.append({
                    '日期': date,
                    '股票代码': code,
                    '开盘': np.random.uniform(10, 100),
                    '最高': np.random.uniform(10, 100),
                    '最低': np.random.uniform(10, 100),
                    '收盘': np.random.uniform(10, 100),
                    '成交量': np.random.uniform(1e6, 1e8),
                    '成交额': np.random.uniform(1e7, 1e9),
                    '振幅': np.random.uniform(0, 10),
                    '涨跌幅': np.random.uniform(-5, 5),
                    '涨跌额': np.random.uniform(-5, 5),
                    '换手率': np.random.uniform(0, 20)
                })
        
        df = pd.DataFrame(data)
        
        # 创建因子数据
        df['momentum_5'] = df['收盘'].pct_change(5)
        df['volatility_5'] = df['收盘'].pct_change().rolling(5).std()
        df['turnover_ratio'] = df['换手率'].rolling(5).mean()
        
        factor_columns = ['momentum_5', 'volatility_5', 'turnover_ratio']
        
        # 测试中性化功能
        df_neutralized = neutralize_factors(df, factor_columns, neutralization_type='both')
        print(f"✓ 成功完成因子中性化")
        
        # 验证中性化后的因子
        for factor in factor_columns:
            assert factor in df_neutralized.columns
            print(f"  {factor}: 均值={df_neutralized[factor].mean():.4f}, 标准差={df_neutralized[factor].std():.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 因子中性化功能测试失败: {e}")
        return False

def test_style_exposure_control():
    """测试风格因子暴露控制功能"""
    print("\n测试风格因子暴露控制功能...")
    try:
        # 创建模拟数据
        np.random.seed(42)
        dates = [datetime(2024, 1, i) for i in range(1, 11)]
        stock_codes = [f"60000{i}" for i in range(1, 11)]
        data = []
        for date in dates:
            for code in stock_codes:
                data.append({
                    '日期': date,
                    '股票代码': code,
                    '开盘': np.random.uniform(10, 100),
                    '最高': np.random.uniform(10, 100),
                    '最低': np.random.uniform(10, 100),
                    '收盘': np.random.uniform(10, 100),
                    '成交量': np.random.uniform(1e6, 1e8),
                    '成交额': np.random.uniform(1e7, 1e9),
                    '振幅': np.random.uniform(0, 10),
                    '涨跌幅': np.random.uniform(-5, 5),
                    '涨跌额': np.random.uniform(-5, 5),
                    '换手率': np.random.uniform(0, 20)
                })
        
        df = pd.DataFrame(data)
        
        # 计算风格因子
        df_style = calculate_style_factors(df)
        
        # 创建模拟的股票评分
        df_style['score'] = np.random.uniform(-2, 2, len(df_style))
        
        # 测试风格暴露控制
        adjusted_scores = control_style_exposure(df_style['score'], 
                                                 df_style[['market_cap', 'valuation', 'growth', 'quality', 'momentum']],
                                                 max_exposure=0.1)
        
        print(f"✓ 成功控制风格因子暴露")
        print(f"  原始评分: 均值={df_style['score'].mean():.2f}, 标准差={df_style['score'].std():.2f}")
        print(f"  调整后评分: 均值={adjusted_scores.mean():.2f}, 标准差={adjusted_scores.std():.2f}")
        
        # 验证调整是否合理
        assert len(adjusted_scores) == len(df_style)
        assert not adjusted_scores.isnull().any()
        
        return True
    except Exception as e:
        print(f"✗ 风格因子暴露控制功能测试失败: {e}")
        return False

def test_risk_control_module():
    """测试风险控制模块的完整流程"""
    print("\n测试风险控制模块完整流程...")
    
    tests = [
        ("行业分类功能", test_industry_classification),
        ("风格因子计算", test_style_factors),
        ("因子中性化", test_factor_neutralization),
        ("风格暴露控制", test_style_exposure_control)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print(f"\n测试结果: 成功 {passed} 个，失败 {failed} 个")
    
    if failed > 0:
        print("\n❌ 风险控制模块存在问题，请检查代码")
        return False
    else:
        print("\n✅ 风险控制模块所有功能测试通过")
        return True

if __name__ == "__main__":
    print("="*70)
    print("风险控制模块功能测试")
    print("="*70)
    
    test_risk_control_module()
