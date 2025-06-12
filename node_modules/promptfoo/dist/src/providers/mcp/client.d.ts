import type { MCPConfig, MCPTool, MCPToolResult } from './types';
export declare class MCPClient {
    private clients;
    private tools;
    private config;
    private transports;
    get hasInitialized(): boolean;
    get connectedServers(): string[];
    constructor(config: MCPConfig);
    initialize(): Promise<void>;
    private connectToServer;
    private getAuthHeaders;
    getAllTools(): MCPTool[];
    callTool(name: string, args: Record<string, unknown>): Promise<MCPToolResult>;
    cleanup(): Promise<void>;
}
//# sourceMappingURL=client.d.ts.map