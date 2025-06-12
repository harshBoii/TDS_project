import type { ApiProvider, ProviderOptions } from '../types';
import type { CallApiContextParams, CallApiOptionsParams, ProviderResponse } from '../types';
import type { EnvOverrides } from '../types/env';
import { OpenAiChatCompletionProvider } from './openai/chat';
import type { OpenAiCompletionOptions } from './openai/types';
/**
 * Calculate the cost of using the Perplexity API based on token usage
 *
 * Pricing based on Perplexity's documentation:
 * https://docs.perplexity.ai/docs/pricing
 *
 * @param modelName - Name of the Perplexity model
 * @param promptTokens - Number of prompt tokens
 * @param completionTokens - Number of completion tokens
 * @param usageTier - Usage tier (high, medium, low) - defaults to medium
 * @returns Cost in USD
 */
export declare function calculatePerplexityCost(modelName: string, promptTokens?: number, completionTokens?: number, usageTier?: 'high' | 'medium' | 'low'): number;
interface PerplexityProviderOptions extends ProviderOptions {
    config?: OpenAiCompletionOptions & {
        usage_tier?: 'high' | 'medium' | 'low';
        search_domain_filter?: string[];
        search_recency_filter?: string;
        return_related_questions?: boolean;
        return_images?: boolean;
        search_after_date_filter?: string;
        search_before_date_filter?: string;
        web_search_options?: {
            search_context_size?: 'low' | 'medium' | 'high';
            user_location?: {
                latitude?: number;
                longitude?: number;
                country?: string;
            };
        };
        [key: string]: any;
    };
}
/**
 * Perplexity API provider
 *
 * Extends the OpenAI chat completion provider to use Perplexity's API endpoint
 * and adds custom cost calculation.
 */
export declare class PerplexityProvider extends OpenAiChatCompletionProvider {
    private usageTier;
    modelName: string;
    config: any;
    constructor(modelName: string, providerOptions?: PerplexityProviderOptions);
    /**
     * Override callApi to use our custom cost calculation
     */
    callApi(prompt: string, context?: CallApiContextParams, callApiOptions?: CallApiOptionsParams): Promise<ProviderResponse>;
    id(): string;
    toString(): string;
    toJSON(): {
        provider: string;
        model: string;
        config: any;
    };
}
/**
 * Creates a Perplexity API provider
 *
 * @param providerPath - Provider path, e.g., "perplexity:sonar"
 * @param options - Provider options
 * @returns A Perplexity API provider
 */
export declare function createPerplexityProvider(providerPath: string, options?: {
    config?: ProviderOptions;
    id?: string;
    env?: EnvOverrides;
}): ApiProvider;
export {};
//# sourceMappingURL=perplexity.d.ts.map