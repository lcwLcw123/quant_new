#!/usr/bin/env python3
"""测试脚本：验证已成功移除量化交易系统中的模拟数据回退机制"""

import sys
import os
import re

def check_file(file_path):
    """检查文件是否包含模拟数据相关代码"""
    print(f"检查文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # 检查是否包含generate_dummy_data函数
    if "generate_dummy_data" in content:
        issues.append("包含generate_dummy_data()函数")
    
    # 检查是否包含模拟数据生成代码
    if "np.random" in content and ("生成" in content or "模拟" in content) and "seed" not in content:
        issues.append("包含模拟数据生成代码")
    
    # 检查是否有模拟数据回退机制
    if "使用模拟数据" in content or "模拟数据进行演示" in content:
        issues.append("包含模拟数据回退机制")
    
    return issues

def main():
    print("="*70)
    print("量化交易系统 - 模拟数据清除检查")
    print("="*70)
    
    # 需要检查的文件列表
    files_to_check = [
        "quant_system.py",
        "quant_system_full.py",
        "quant_lightweight.py"
    ]
    
    all_issues = []
    
    for filename in files_to_check:
        file_path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(file_path):
            issues = check_file(file_path)
            if issues:
                print(f"❌ {filename} 发现问题:")
                for issue in issues:
                    print(f"   - {issue}")
                all_issues.extend([f"{filename}: {issue}" for issue in issues])
            else:
                print(f"✅ {filename} 检查通过")
        else:
            print(f"⚠️  文件不存在: {filename}")
    
    print()
    print("="*70)
    if all_issues:
        print(f"❌ 发现 {len(all_issues)} 个问题需要修复")
        sys.exit(1)
    else:
        print("✅ 所有量化交易系统文件已成功清除模拟数据机制")
        sys.exit(0)

if __name__ == "__main__":
    main()
