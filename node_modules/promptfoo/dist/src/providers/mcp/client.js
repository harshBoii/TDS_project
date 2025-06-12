"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MCPClient = void 0;
const index_js_1 = require("@modelcontextprotocol/sdk/client/index.js");
const logger_1 = __importDefault(require("../../logger"));
class MCPClient {
    get hasInitialized() {
        return this.clients.size > 0;
    }
    get connectedServers() {
        return Array.from(this.clients.keys());
    }
    constructor(config) {
        this.clients = new Map();
        this.tools = new Map();
        this.transports = new Map();
        this.config = config;
    }
    async initialize() {
        if (!this.config.enabled) {
            return;
        }
        // Initialize servers
        const servers = this.config.servers || (this.config.server ? [this.config.server] : []);
        for (const server of servers) {
            await this.connectToServer(server);
        }
    }
    async connectToServer(server) {
        const serverKey = server.name || server.url || server.path || 'default';
        const client = new index_js_1.Client({ name: 'promptfoo-MCP', version: '1.0.0' });
        let transport;
        try {
            if (server.command && server.args) {
                const { StdioClientTransport } = await Promise.resolve().then(() => __importStar(require('@modelcontextprotocol/sdk/client/stdio.js')));
                // NPM package or other command execution
                transport = new StdioClientTransport({
                    command: server.command,
                    args: server.args,
                });
                await client.connect(transport);
            }
            else if (server.path) {
                // Local server file
                const isJs = server.path.endsWith('.js');
                const isPy = server.path.endsWith('.py');
                if (!isJs && !isPy) {
                    throw new Error('Local server must be a .js or .py file');
                }
                const command = isPy
                    ? process.platform === 'win32'
                        ? 'python'
                        : 'python3'
                    : process.execPath;
                const { StdioClientTransport } = await Promise.resolve().then(() => __importStar(require('@modelcontextprotocol/sdk/client/stdio.js')));
                transport = new StdioClientTransport({
                    command,
                    args: [server.path],
                });
                await client.connect(transport);
            }
            else if (server.url) {
                // Get auth headers and combine with custom headers
                const authHeaders = this.getAuthHeaders(server);
                const headers = {
                    ...(server.headers || {}),
                    ...authHeaders,
                };
                // Only set options if we have headers
                const options = Object.keys(headers).length > 0 ? { requestInit: { headers } } : undefined;
                try {
                    const { StreamableHTTPClientTransport } = await Promise.resolve().then(() => __importStar(require('@modelcontextprotocol/sdk/client/streamableHttp.js')));
                    transport = new StreamableHTTPClientTransport(new URL(server.url), options);
                    await client.connect(transport);
                    logger_1.default.debug('Connected using Streamable HTTP transport');
                }
                catch (error) {
                    logger_1.default.error(`Failed to connect to MCP server ${serverKey}: ${error}`);
                    const { SSEClientTransport } = await Promise.resolve().then(() => __importStar(require('@modelcontextprotocol/sdk/client/sse.js')));
                    transport = new SSEClientTransport(new URL(server.url), options);
                    await client.connect(transport);
                    logger_1.default.debug('Connected using SSE transport');
                }
            }
            else {
                throw new Error('Either command+args or path or url must be specified for MCP server');
            }
            // List available tools
            const toolsResult = await client.listTools();
            const serverTools = toolsResult?.tools?.map((tool) => ({
                name: tool.name,
                description: tool.description || '',
                inputSchema: tool.inputSchema,
            })) || [];
            // Filter tools if specified
            let filteredTools = serverTools;
            if (this.config.tools) {
                filteredTools = serverTools.filter((tool) => this.config.tools?.includes(tool.name));
            }
            if (this.config.exclude_tools) {
                filteredTools = filteredTools.filter((tool) => !this.config.exclude_tools?.includes(tool.name));
            }
            this.transports.set(serverKey, transport);
            this.clients.set(serverKey, client);
            this.tools.set(serverKey, filteredTools);
            if (this.config.verbose) {
                console.log(`Connected to MCP server ${serverKey} with tools:`, filteredTools.map((tool) => tool.name));
            }
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            if (this.config.debug) {
                logger_1.default.error(`Failed to connect to MCP server ${serverKey}: ${errorMessage}`);
            }
            throw new Error(`Failed to connect to MCP server ${serverKey}: ${errorMessage}`);
        }
    }
    getAuthHeaders(server) {
        if (!server.auth) {
            return {};
        }
        if (server.auth.type === 'bearer' && server.auth.token) {
            return { Authorization: `Bearer ${server.auth.token}` };
        }
        if (server.auth.type === 'api_key' && server.auth.api_key) {
            return { 'X-API-Key': server.auth.api_key };
        }
        return {};
    }
    getAllTools() {
        return Array.from(this.tools.values()).flat();
    }
    async callTool(name, args) {
        // Find which server has this tool
        for (const [serverKey, client] of this.clients.entries()) {
            const serverTools = this.tools.get(serverKey) || [];
            if (serverTools.some((tool) => tool.name === name)) {
                try {
                    const result = await client.callTool({ name, arguments: args });
                    return {
                        content: result?.content?.toString() || '',
                    };
                }
                catch (error) {
                    const errorMessage = error instanceof Error ? error.message : String(error);
                    if (this.config.debug) {
                        logger_1.default.error(`Error calling tool ${name}: ${errorMessage}`);
                    }
                    return {
                        content: '',
                        error: errorMessage,
                    };
                }
            }
        }
        throw new Error(`Tool ${name} not found in any connected MCP server`);
    }
    async cleanup() {
        for (const [serverKey, client] of this.clients.entries()) {
            try {
                const transport = this.transports.get(serverKey);
                if (transport) {
                    await transport.close();
                }
                await client.close();
            }
            catch (error) {
                if (this.config.debug) {
                    logger_1.default.error(`Error during cleanup: ${error instanceof Error ? error.message : String(error)}`);
                }
            }
        }
        this.clients.clear();
        this.transports.clear();
        this.tools.clear();
    }
}
exports.MCPClient = MCPClient;
//# sourceMappingURL=client.js.map