#!/usr/bin/env python3
"""
本地 Tavily MCP 服务器
模拟 MCP 协议，直接使用 Tavily API
"""

import json
import sys
import subprocess
from typing import Dict, Any

# Tavily API 配置
API_KEY = "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"

class TavilyMCPServer:
    """模拟 MCP 服务器的本地进程"""
    
    def __init__(self):
        self.tools = {
            "tavily_search": {
                "name": "tavily_search",
                "description": "使用 Tavily API 搜索网络",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查询"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "最大结果数量",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """执行搜索"""
        try:
            import requests
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": API_KEY,
                "query": query,
                "max_results": max_results,
                "include_answer": True
            }
            
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            # 格式化结果
            formatted = {
                "answer": results.get("answer", ""),
                "results": []
            }
            
            if "results" in results:
                for result in results["results"]:
                    formatted["results"].append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0)
                    })
            
            return formatted
        except Exception as e:
            return {"error": str(e)}
    
    def run(self):
        """运行 MCP 服务器循环"""
        print("🚀 本地 Tavily MCP 服务器启动...")
        print(f"📊 可用工具: {list(self.tools.keys())}")
        print("⌨️  输入 'search <query>' 进行搜索")
        print("⌨️  输入 'quit' 退出")
        print("=" * 50)
        
        while True:
            try:
                # 从标准输入读取命令
                line = sys.stdin.readline().strip()
                
                if not line:
                    break
                
                if line.lower() == 'quit':
                    print("👋 服务器退出")
                    break
                
                if line.startswith('search '):
                    query = line[7:].strip()
                    print(f"\n🔍 搜索: {query}")
                    
                    results = self.search(query)
                    
                    if "error" in results:
                        print(f"❌ 搜索失败: {results['error']}")
                    else:
                        if results.get("answer"):
                            print(f"🤖 AI 答案:\n   {results['answer']}")
                        
                        print(f"\n📊 找到 {len(results['results'])} 条结果:")
                        for i, result in enumerate(results["results"], 1):
                            print(f"\n{i}. {result['title']}")
                            print(f"   🔗 {result['url']}")
                            if result.get("content"):
                                print(f"   📝 {result['content'][:100]}...")
                            print(f"   ⭐ 相关度: {result['score']:.2f}")
                    print("\n" + "=" * 50)
                
                else:
                    print(f"❓ 未知命令: {line}")
                    print("💡 可用命令: search <query>, quit")
                    
            except KeyboardInterrupt:
                print("\n\n👋 服务器退出")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


def main():
    """主函数"""
    print("🛠️  启动本地 Tavily MCP 服务器...")
    print("💡 这个脚本模拟 MCP 服务器，提供搜索功能")
    print("")
    
    # 检查 requests 库
    try:
        import requests
    except ImportError:
        print("❌ 缺少 requests 库")
        print("💡 安装: pip3 install requests")
        sys.exit(1)
    
    # 创建并启动服务器
    server = TavilyMCPServer()
    server.run()


if __name__ == "__main__":
    main()
