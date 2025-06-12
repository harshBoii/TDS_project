import { type ApiProvider, type AtomicTestCase, type CallApiContextParams, type CallApiOptionsParams, type GuardrailResponse, type NunjucksFilterMap, type Prompt } from '../../types';
interface IterativeMetadata {
    finalIteration: number;
    highestScore: number;
    redteamFinalPrompt?: string;
    redteamHistory: {
        prompt: string;
        output: string;
        score: number;
        isOnTopic: boolean;
        graderPassed: boolean | undefined;
        guardrails: GuardrailResponse | undefined;
    }[];
}
interface TokenUsage {
    total: number;
    prompt: number;
    completion: number;
    numRequests: number;
    cached: number;
}
export declare function runRedteamConversation({ context, filters, injectVar, numIterations, options, prompt, redteamProvider, gradingProvider, targetProvider, test, vars, excludeTargetOutputFromAgenticAttackGeneration, }: {
    context?: CallApiContextParams;
    filters: NunjucksFilterMap | undefined;
    injectVar: string;
    numIterations: number;
    options?: CallApiOptionsParams;
    prompt: Prompt;
    redteamProvider: ApiProvider;
    gradingProvider: ApiProvider;
    targetProvider: ApiProvider;
    test?: AtomicTestCase;
    vars: Record<string, string | object>;
    excludeTargetOutputFromAgenticAttackGeneration: boolean;
}): Promise<{
    output: string;
    metadata: IterativeMetadata;
    tokenUsage: TokenUsage;
}>;
declare class RedteamIterativeProvider implements ApiProvider {
    readonly config: Record<string, string | object>;
    private readonly redteamProvider;
    private readonly injectVar;
    private readonly numIterations;
    private readonly excludeTargetOutputFromAgenticAttackGeneration;
    private readonly gradingProvider;
    constructor(config: Record<string, string | object>);
    id(): string;
    callApi(prompt: string, context?: CallApiContextParams, options?: CallApiOptionsParams): Promise<{
        output: string;
        metadata: IterativeMetadata;
        tokenUsage: TokenUsage;
    }>;
}
export default RedteamIterativeProvider;
//# sourceMappingURL=iterative.d.ts.map