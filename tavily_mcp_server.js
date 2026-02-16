#!/usr/bin/env node

/**
 * Tavily MCP 服务器
 * 实现标准的 MCP 协议
 */

const https = require('https');

// Tavily API 配置
const API_KEY = process.env.TAVILY_API_KEY || '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';
const API_URL = 'https://api.tavily.com/search';

// MCP 工具定义
const TOOLS = {
    'tavily_search': {
        name: 'tavily_search',
        description: '使用 Tavily AI 搜索引擎搜索网络内容',
        inputSchema: {
            type: 'object',
            properties: {
                query: {
                    type: 'string',
                    description: '搜索查询字符串'
                },
                max_results: {
                    type: 'number',
                    description: '返回结果的最大数量',
                    default: 10
                },
                search_depth: {
                    type: 'string',
                    description: '搜索深度：basic 或 advanced',
                    enum: ['basic', 'advanced'],
                    default: 'basic'
                },
                include_images: {
                    type: 'boolean',
                    description: '是否包含图像结果',
                    default: false
                },
                include_answer: {
                    type: 'boolean',
                    description: '是否包含 AI 生成的答案',
                    default: true
                }
            },
            required: ['query']
        }
    }
};

// 执行 Tavily 搜索
async function tavilySearch(params) {
    const {
        query,
        max_results = 10,
        search_depth = 'basic',
        include_images = false,
        include_answer = true
    } = params;

    return new Promise((resolve, reject) => {
        const data = JSON.stringify({
            api_key: API_KEY,
            query: query,
            max_results: max_results,
            search_depth: search_depth,
            include_images: include_images,
            include_answer: include_answer
        });

        const options = {
            hostname: 'api.tavily.com',
            path: '/search',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data)
            }
        };

        const req = https.request(options, (res) => {
            let responseData = '';

            res.on('data', (chunk) => {
                responseData += chunk;
            });

            res.on('end', () => {
                try {
                    const result = JSON.parse(responseData);
                    resolve(result);
                } catch (error) {
                    reject({ error: 'JSON解析失败: ' + error.message });
                }
            });
        });

        req.on('error', (error) => {
            reject({ error: '请求失败: ' + error.message });
        });

        req.write(data);
        req.end();
    });
}

// 发送 MCP 响应
function sendResponse(response) {
    process.stdout.write(JSON.stringify(response) + '\n');
}

// 发送错误响应
function sendError(id, error) {
    sendResponse({
        jsonrpc: '2.0',
        id: id,
        error: {
            code: -32000,
            message: error.message || '未知错误',
            data: error
        }
    });
}

// 处理工具调用
async function handleToolCall(id, params) {
    const { name, arguments: args } = params;

    try {
        if (name === 'tavily_search') {
            const result = await tavilySearch(args);
            sendResponse({
                jsonrpc: '2.0',
                id: id,
                result: {
                    content: [{
                        type: 'text',
                        text: formatSearchResult(result)
                    }]
                }
            });
        } else {
            throw new Error(`未知工具: ${name}`);
        }
    } catch (error) {
        sendError(id, error);
    }
}

// 格式化搜索结果
function formatSearchResult(result) {
    let output = '';

    if (result.error) {
        return `❌ 搜索失败: ${result.error}`;
    }

    // AI 答案
    if (result.answer) {
        output += `🤖 **AI 答案**:\n${result.answer}\n\n`;
    }

    // 搜索结果
    if (result.results && result.results.length > 0) {
        output += `📊 **找到 ${result.results.length} 条结果**:\n\n`;

        result.results.forEach((item, index) => {
            output += `${index + 1}. **${item.title}**\n`;
            output += `   🔗 ${item.url}\n`;
            if (item.content) {
                output += `   📝 ${item.content.substring(0, 100)}...\n`;
            }
            output += `   ⭐ 相关度: ${item.score.toFixed(2)}\n\n`;
        });
    } else {
        output += '📭 未找到相关结果';
    }

    return output;
}

// 主循环
function main() {
    // 发送初始化消息
    sendResponse({
        jsonrpc: '2.0',
        method: 'initialize',
        params: {
            protocolVersion: '2024-11-05',
            capabilities: {
                tools: {}
            },
            serverInfo: {
                name: 'tavily-mcp-server',
                version: '1.0.0'
            }
        }
    });

    // 监听标准输入
    let buffer = '';
    process.stdin.on('data', (data) => {
        buffer += data.toString();

        // 处理完整的 JSON-RPC 消息
        const lines = buffer.split('\n');
        for (let i = 0; i < lines.length - 1; i++) {
            try {
                const request = JSON.parse(lines[i]);
                handleRequest(request);
            } catch (error) {
                console.error(`解析错误: ${error.message}`);
            }
        }
        buffer = lines[lines.length - 1];
    });

    function handleRequest(request) {
        if (request.jsonrpc !== '2.0') {
            return;
        }

        switch (request.method) {
            case 'tools/list':
                sendResponse({
                    jsonrpc: '2.0',
                    id: request.id,
                    result: {
                        tools: Object.values(TOOLS)
                    }
                });
                break;

            case 'tools/call':
                handleToolCall(request.id, request.params);
                break;

            case 'initialize':
                sendResponse({
                    jsonrpc: '2.0',
                    id: request.id,
                    result: {
                        protocolVersion: '2024-11-05',
                        capabilities: {
                            tools: {}
                        },
                        serverInfo: {
                            name: 'tavily-mcp-server',
                            version: '1.0.0'
                        }
                    }
                });
                break;

            default:
                if (request.id) {
                    sendError(request.id, { message: '未知方法' });
                }
        }
    }
}

// 启动服务器
console.error('🚀 Tavily MCP 服务器启动...');
console.error(`📊 可用工具: ${Object.keys(TOOLS).join(', ')}`);
console.error('📝 正在监听 JSON-RPC 请求...');
console.error('='.repeat(50));

main();
