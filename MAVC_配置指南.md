# Tavily MCP 配置完整指南

## 问题诊断
当前主要问题：npm 访问令牌过期，导致无法从 npm registry 安装包。

## 🔧 解决方案

### 方案 1: 修复 npm 访问令牌（推荐）

```bash
# 1. 登录 npm
npm login

# 2. 或者清理 npm 缓存和重新设置
npm cache clean --force
npm config set registry https://registry.npmjs.org/

# 3. 安装 Tavily MCP Server
npx -y @tavily/mcp-server --version

# 4. 如果上面成功，设置环境变量
export TAVILY_API_KEY="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"

# 5. 创建配置文件
mkdir -p ~/.config/claude
cat > ~/.config/claude/claude_desktop_config.json << 'EOF'
{
  "mcpServers": {
    "tavily": {
      "command": "npx",
      "args": ["-y", "@tavily/mcp-server"],
      "env": {
        "TAVILY_API_KEY": "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"
      }
    }
  }
}
EOF

# 6. 重启 Claude Desktop
```

### 方案 2: 使用 GitHub 直接安装

```bash
# 1. 克隆 Tavily MCP Server 仓库
git clone https://github.com/Tavily/tavily-mcp-server.git ~/tavily-mcp-server

# 2. 进入目录
cd ~/tavily-mcp-server

# 3. 安装依赖
npm install

# 4. 创建配置文件
cat > ~/.config/claude/claude_desktop_config.json << 'EOF'
{
  "mcpServers": {
    "tavily": {
      "command": "node",
      "args": ["/Users/hanyinghui/tavily-mcp-server/src/index.js"],
      "env": {
        "TAVILY_API_KEY": "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"
      }
    }
  }
}
EOF

# 5. 重启 Claude Desktop
```

### 方案 3: 使用 Docker（如果可用）

```bash
# 1. 使用 Docker 运行 Tavily MCP Server
docker run -d \
  -e TAVILY_API_KEY="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" \
  -p 3000:3000 \
  tavily/mcp-server

# 2. 创建配置文件连接到 Docker 容器
cat > ~/.config/claude/claude_desktop_config.json << 'EOF'
{
  "mcpServers": {
    "tavily": {
      "command": "docker",
      "args": ["run", "-e", "TAVILY_API_KEY=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", "tavily/mcp-server"],
      "env": {
        "TAVILY_API_KEY": "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"
      }
    }
  }
}
EOF

# 3. 重启 Claude Desktop
```

### 方案 4: 手动下载和安装

```bash
# 1. 从 GitHub 下载最新的 release
wget https://github.com/Tavily/tavily-mcp-server/releases/latest/download/tavily-mcp-server.tar.gz

# 2. 解压
tar -xzf tavily-mcp-server.tar.gz

# 3. 进入目录并安装
cd tavily-mcp-server
npm install

# 4. 运行
node src/index.js
```

## ✅ 验证配置

### 1. 检查配置文件
```bash
cat ~/.config/claude/claude_desktop_config.json
```

### 2. 检查 MCP 服务器是否运行
```bash
# 如果使用 npx
npx -y @tavily/mcp-server --version

# 或者检查进程
ps aux | grep -i "tavily\|mcp"
```

### 3. 测试 API 密钥
```bash
curl -X POST https://api.tavily.com/search \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "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",
    "query": "test",
    "max_results": 1
  }'
```

## 🔍 故障排除

### 问题 1: npm 访问令牌过期
```bash
# 解决：重新登录或清理缓存
npm logout
npm login
```

### 问题 2: 找不到配置文件
```bash
# 解决：确保配置文件在正确位置
ls -la ~/.config/claude/
```

### 问题 3: MCP 服务器无法启动
```bash
# 解决：检查端口占用和权限
lsof -i :3000
```

### 问题 4: Claude Desktop 不识别 MCP
```bash
# 解决：完全重启 Claude Desktop
killall "Claude Desktop"
open -a "Claude Desktop"
```

## 📋 配置完成后的验证清单

- [ ] npm 访问令牌已修复
- [ ] Tavily MCP Server 已安装
- [ ] 配置文件已创建在 ~/.config/claude/
- [ ] API 密钥已正确设置
- [ ] Claude Desktop 已重启
- [ ] 可以在 Claude 中使用 Tavily 搜索

## 📞 需要帮助？

如果以上方案都无法解决问题，请：
1. 检查 Claude Desktop 日志
2. 查看 npm 错误日志：`cat ~/.npm/_logs/$(ls -t ~/.npm/_logs | head -1)/*.log`
3. 访问 Tavily 官方文档：https://docs.tavily.com/docs/mcp
