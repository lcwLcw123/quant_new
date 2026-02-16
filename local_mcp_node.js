const https = require('https');

// Tavily API 配置
const API_KEY = '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';
const API_URL = 'https://api.tavily.com/search';

// 模拟 MCP 服务器
class LocalMCPServer {
    constructor() {
        this.tools = {
            'tavily_search': {
                name: 'tavily_search',
                description: '使用 Tavily API 搜索网络',
                inputSchema: {
                    type: 'object',
                    properties: {
                        query: {
                            type: 'string',
                            description: '搜索查询'
                        },
                        max_results: {
                            type: 'integer',
                            description: '最大结果数量',
                            default: 10
                        }
                    },
                    required: ['query']
                }
            }
        };
    }

    // 搜索功能
    search(query, maxResults = 10) {
        return new Promise((resolve, reject) => {
            const data = JSON.stringify({
                api_key: API_KEY,
                query: query,
                max_results: maxResults,
                include_answer: true
            });

            const options = {
                hostname: 'api.tavily.com',
                path: '/search',
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': data.length
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
                        
                        // 格式化结果
                        const formatted = {
                            answer: result.answer || '',
                            results: []
                        };

                        if (result.results) {
                            result.results.forEach(r => {
                                formatted.results.push({
                                    title: r.title || '',
                                    url: r.url || '',
                                    content: r.content || '',
                                    score: r.score || 0
                                });
                            });
                        }

                        resolve(formatted);
                    } catch (error) {
                        reject({ error: '解析响应失败: ' + error.message });
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

    // 列出可用工具
    listTools() {
        return this.tools;
    }

    // 格式化输出
    formatOutput(results) {
        let output = '';
        
        if (results.error) {
            output += '❌ 搜索失败: ' + results.error + '\n';
        } else {
            if (results.answer) {
                output += '🤖 **AI 答案**:\n';
                output += '   ' + results.answer + '\n\n';
            }

            if (results.results && results.results.length > 0) {
                output += '📊 **找到 ' + results.results.length + ' 条结果**:\n\n';

                results.results.forEach((result, index) => {
                    output += (index + 1) + '. **' + result.title + '**\n';
                    output += '   🔗 ' + result.url + '\n';
                    if (result.content) {
                        output += '   📝 ' + result.content.substring(0, 100) + '...\n';
                    }
                    output += '   ⭐ 相关度: ' + result.score.toFixed(2) + '\n\n';
                });
            }
        }

        return output;
    }

    // 运行交互模式
    async runInteractive() {
        console.log('🚀 本地 Tavily MCP 服务器启动');
        console.log('📊 可用工具: ' + Object.keys(this.tools).join(', '));
        console.log('⌨️  输入 "search <query>" 进行搜索');
        console.log('⌨️  输入 "quit" 退出');
        console.log('='.repeat(50));
        console.log('');

        const readline = require('readline');
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });

        const ask = (query) => {
            return new Promise((resolve) => {
                rl.question(query, resolve);
            });
        };

        while (true) {
            try {
                const input = (await ask('> ')).trim();

                if (!input) {
                    continue;
                }

                if (input.toLowerCase() === 'quit') {
                    console.log('👋 服务器退出');
                    break;
                }

                if (input.startsWith('search ')) {
                    const query = input.substring(7).trim();
                    console.log('\n🔍 搜索: ' + query);

                    const results = await this.search(query);
                    console.log(this.formatOutput(results));
                    console.log('='.repeat(50));
                    console.log('');
                } else {
                    console.log('❓ 未知命令: ' + input);
                    console.log('💡 可用命令: search <query>, quit');
                    console.log('');
                }
            } catch (error) {
                console.log('❌ 错误: ' + error.message);
                console.log('');
            }
        }

        rl.close();
    }
}

// 主函数
async function main() {
    console.log('🛠️  启动本地 Tavily MCP 服务器');
    console.log('💡 这个脚本模拟 MCP 服务器，提供搜索功能');
    console.log('');

    const server = new LocalMCPServer();

    // 检查命令行参数
    const args = process.argv.slice(2);
    if (args.length > 0) {
        const command = args[0];
        
        if (command === 'search') {
            if (args.length > 1) {
                const query = args.slice(1).join(' ');
                console.log('🔍 搜索: ' + query);
                
                const results = await server.search(query);
                console.log(server.formatOutput(results));
            } else {
                console.log('❌ 请提供搜索查询');
                console.log('💡 用法: node local_mcp_node.js search "查询内容"');
            }
        } else {
            console.log('❓ 未知命令: ' + command);
            console.log('💡 可用命令: search <query>');
        }
    } else {
        // 没有参数，运行交互模式
        await server.runInteractive();
    }
}

// 启动服务器
if (require.main === module) {
    main().catch(error => {
        console.error('❌ 启动失败:', error);
        process.exit(1);
    });
}
