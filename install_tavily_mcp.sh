#!/bin/bash

# Tavily MCP 自动安装和配置脚本

set -e

echo "🚀 Tavily MCP 自动安装和配置脚本"
echo "================================"

# API 密钥
API_KEY="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"

echo "1️⃣  检查环境..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装，请先安装 npm"
    exit 1
fi

echo "✅ Node.js 和 npm 已安装"
echo "   Node: $(node -v)"
echo "   npm: $(npm -v)"

echo ""
echo "2️⃣  尝试方案 1: 修复 npm 并安装..."

# 尝试修复 npm
npm logout 2>/dev/null || true
npm cache clean --force
npm config set registry https://registry.npmjs.org/

# 尝试使用 npx 安装
if npx -y @tavily/mcp-server --version 2>/dev/null; then
    echo "✅ 使用 npx 安装成功！"
    echo "   $(npx -y @tavily/mcp-server --version)"
    
    # 创建配置文件
    mkdir -p ~/.config/claude
    cat > ~/.config/claude/claude_desktop_config.json << EOF
{
  "mcpServers": {
    "tavily": {
      "command": "npx",
      "args": ["-y", "@tavily/mcp-server"],
      "env": {
        "TAVILY_API_KEY": "$API_KEY"
      }
    }
  }
}
EOF
    
    echo "✅ 配置文件已创建"
    echo "📝 配置文件位置: ~/.config/claude/claude_desktop_config.json"
    echo ""
    echo "🎉 安装完成！"
    echo "   请重启 Claude Desktop 以加载新配置"
    exit 0
fi

echo "⚠️  npx 安装失败，尝试方案 2..."
echo ""

echo "3️⃣  尝试方案 2: 从 GitHub 克隆..."

# 检查 git
if ! command -v git &> /dev/null; then
    echo "❌ Git 未安装"
    exit 1
fi

# 尝试克隆
MCP_DIR="$HOME/tavily-mcp-server"
if [ -d "$MCP_DIR" ]; then
    echo "📁 目录已存在，尝试更新..."
    cd "$MCP_DIR" && git pull || true
else
    echo "📁 克隆 Tavily MCP Server..."
    git clone https://github.com/tavily/tavily-mcp-server.git "$MCP_DIR"
fi

if [ -d "$MCP_DIR" ]; then
    echo "✅ 仓库克隆成功"
    
    # 安装依赖
    echo "📦 安装依赖..."
    cd "$MCP_DIR"
    npm install
    
    # 创建配置文件
    mkdir -p ~/.config/claude
    cat > ~/.config/claude/claude_desktop_config.json << EOF
{
  "mcpServers": {
    "tavily": {
      "command": "node",
      "args": ["$MCP_DIR/src/index.js"],
      "env": {
        "TAVILY_API_KEY": "$API_KEY"
      }
    }
  }
}
EOF
    
    echo "✅ 配置文件已创建"
    echo "📝 配置文件位置: ~/.config/claude/claude_desktop_config.json"
    echo ""
    echo "🎉 安装完成！"
    echo "   请重启 Claude Desktop 以加载新配置"
    exit 0
fi

echo "❌ GitHub 克隆失败，尝试方案 3..."
echo ""

echo "4️⃣  尝试方案 3: 使用淘宝镜像..."

# 切换到淘宝镜像
npm config set registry https://registry.npmmirror.com/

# 尝试从淘宝镜像安装（如果有的话）
if npx -y @tavily/mcp-server --version 2>/dev/null; then
    echo "✅ 从淘宝镜像安装成功！"
    
    # 创建配置文件
    mkdir -p ~/.config/claude
    cat > ~/.config/claude/claude_desktop_config.json << EOF
{
  "mcpServers": {
    "tavily": {
      "command": "npx",
      "args": ["-y", "@tavily/mcp-server"],
      "env": {
        "TAVILY_API_KEY": "$API_KEY"
      }
    }
  }
}
EOF
    
    echo "✅ 配置文件已创建"
    exit 0
fi

echo ""
echo "❌ 所有自动安装方案都失败了"
echo ""
echo "📋 请手动执行以下步骤："
echo "   1. 重新登录 npm: npm login"
echo "   2. 尝试手动安装: npx -y @tavily/mcp-server"
echo "   3. 或者从 GitHub 克隆: git clone https://github.com/tavily/tavily-mcp-server.git"
echo "   4. 查看详细指南: cat quant_new/MAVC_配置指南.md"
echo ""
echo "💬 需要帮助？请查看配置指南或联系技术支持"
