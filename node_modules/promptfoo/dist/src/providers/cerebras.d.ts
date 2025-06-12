import type { ApiProvider, ProviderOptions } from '../types';
import type { EnvOverrides } from '../types/env';
/**
 * Creates a Cerebras provider using OpenAI-compatible chat endpoints
 *
 * Documentation: https://docs.cerebras.ai
 *
 * Cerebras API supports the OpenAI-compatible chat completion interface.
 * All parameters are automatically passed through to the Cerebras API.
 */
export declare function createCerebrasProvider(providerPath: string, options?: {
    config?: ProviderOptions;
    id?: string;
    env?: EnvOverrides;
}): ApiProvider;
//# sourceMappingURL=cerebras.d.ts.map