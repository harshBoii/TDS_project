import type { ApiProvider, CallApiContextParams, CallApiOptionsParams, ProviderOptions, ProviderResponse } from '../../../types/providers';
export declare class MemoryPoisoningProvider implements ApiProvider {
    readonly config: ProviderOptions;
    constructor(config: ProviderOptions);
    id(): string;
    /**
     * Generates a scenario containing a memory and a follow up question/command which is dependent on the memory.
     *
     *  1. Generate a scenario containing memories and follow up questions/commands which are dependent on the memories.
     *  2. Send the memory message to the target.
     *  3. Send the test case to the target; the test case should poison the memory created in the previous step.
     *  4. Send the follow up question to the target.
     *  5. A memory is successfully poisoned if the follow up question is answered incorrectly.
     *
     * @param prompt The poisoned prompt.
     * @param context The context of the poisoned prompt.
     * @param options The options for the poisoned prompt.
     */
    callApi(prompt: string, context?: CallApiContextParams, options?: CallApiOptionsParams): Promise<ProviderResponse>;
}
//# sourceMappingURL=memoryPoisoning.d.ts.map