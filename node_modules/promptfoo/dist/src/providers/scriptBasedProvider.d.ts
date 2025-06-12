import type { LoadApiProviderContext } from '../types';
import type { ApiProvider, ProviderOptions } from '../types/providers';
/**
 * Creates a factory for script-based providers (exec, golang, python)
 * @param prefix The provider prefix to match (e.g. 'exec:')
 * @param fileExtension Optional file extension to match when using file:// format
 * @param providerConstructor The provider class constructor
 * @returns A ProviderFactory for the specified script-based provider
 */
export declare function createScriptBasedProviderFactory(prefix: string, fileExtension: string | null, providerConstructor: new (scriptPath: string, options: ProviderOptions) => ApiProvider): {
    test: (providerPath: string) => boolean;
    create: (providerPath: string, providerOptions: ProviderOptions, context: LoadApiProviderContext) => Promise<ApiProvider>;
};
//# sourceMappingURL=scriptBasedProvider.d.ts.map