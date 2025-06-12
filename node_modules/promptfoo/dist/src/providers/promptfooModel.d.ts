import type { CallApiContextParams, CallApiOptionsParams, ProviderOptions, ProviderResponse } from '../types/providers';
import type { ApiProvider } from '../types/providers';
interface PromptfooModelOptions extends ProviderOptions {
    model: string;
    config?: Record<string, any>;
}
/**
 * Provider that connects to the PromptfooModel task of the server.
 */
export declare class PromptfooModelProvider implements ApiProvider {
    private readonly model;
    readonly config: Record<string, any>;
    constructor(model: string, options?: PromptfooModelOptions);
    id(): string;
    callApi(prompt: string, context?: CallApiContextParams, options?: CallApiOptionsParams): Promise<ProviderResponse>;
}
export {};
//# sourceMappingURL=promptfooModel.d.ts.map