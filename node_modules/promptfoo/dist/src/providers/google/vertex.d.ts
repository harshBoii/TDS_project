import type { ApiEmbeddingProvider, ApiProvider, CallApiContextParams, ProviderResponse, ProviderEmbeddingResponse } from '../../types';
import type { EnvOverrides } from '../../types/env';
import type { CompletionOptions } from './types';
declare class VertexGenericProvider implements ApiProvider {
    modelName: string;
    config: CompletionOptions;
    env?: EnvOverrides;
    constructor(modelName: string, options?: {
        config?: CompletionOptions;
        id?: string;
        env?: EnvOverrides;
    });
    id(): string;
    toString(): string;
    getApiHost(): string | undefined;
    getProjectId(): Promise<string | undefined>;
    getApiKey(): string | undefined;
    getRegion(): string;
    getPublisher(): string | undefined;
    getApiVersion(): string;
    callApi(prompt: string): Promise<ProviderResponse>;
}
export declare class VertexChatProvider extends VertexGenericProvider {
    private mcpClient;
    private initializationPromise;
    constructor(modelName: string, options?: {
        config?: CompletionOptions;
        id?: string;
        env?: EnvOverrides;
    });
    private initializeMCP;
    callApi(prompt: string, context?: CallApiContextParams): Promise<ProviderResponse>;
    callClaudeApi(prompt: string, context?: CallApiContextParams): Promise<ProviderResponse>;
    callGeminiApi(prompt: string, context?: CallApiContextParams): Promise<ProviderResponse>;
    callPalm2Api(prompt: string): Promise<ProviderResponse>;
    callLlamaApi(prompt: string, context?: CallApiContextParams): Promise<ProviderResponse>;
    cleanup(): Promise<void>;
}
export declare class VertexEmbeddingProvider implements ApiEmbeddingProvider {
    modelName: string;
    config: any;
    env?: any;
    constructor(modelName: string, config?: any, env?: any);
    id(): string;
    getRegion(): string;
    getApiVersion(): string;
    callApi(): Promise<ProviderResponse>;
    callEmbeddingApi(input: string): Promise<ProviderEmbeddingResponse>;
}
export declare const DefaultGradingProvider: VertexChatProvider;
export declare const DefaultEmbeddingProvider: VertexEmbeddingProvider;
export {};
//# sourceMappingURL=vertex.d.ts.map