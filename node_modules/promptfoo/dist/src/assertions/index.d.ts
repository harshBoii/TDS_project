import { matchesAnswerRelevance, matchesClassification, matchesClosedQa, matchesContextFaithfulness, matchesContextRecall, matchesContextRelevance, matchesFactuality, matchesLlmRubric, matchesModeration, matchesSelectBest, matchesSimilarity } from '../matchers';
import type { ProviderResponse, ScoringFunction } from '../types';
import { type ApiProvider, type Assertion, type AtomicTestCase, type GradingResult } from '../types';
export declare const MODEL_GRADED_ASSERTION_TYPES: Set<"moderation" | `promptfoo:redteam:${string}` | "cost" | "factuality" | "answer-relevance" | "bleu" | "classifier" | "contains" | "contains-all" | "contains-any" | "contains-json" | "contains-sql" | "contains-xml" | "context-faithfulness" | "context-recall" | "context-relevance" | "equals" | "g-eval" | "gleu" | "guardrails" | "icontains" | "icontains-all" | "icontains-any" | "is-json" | "is-refusal" | "is-sql" | "is-valid-function-call" | "is-valid-openai-function-call" | "is-valid-openai-tools-call" | "is-xml" | "javascript" | "latency" | "levenshtein" | "llm-rubric" | "pi" | "meteor" | "model-graded-closedqa" | "model-graded-factuality" | "perplexity" | "perplexity-score" | "python" | "regex" | "rouge-n" | "similar" | "starts-with" | "webhook" | "not-moderation" | "not-cost" | "not-factuality" | "not-answer-relevance" | "not-bleu" | "not-classifier" | "not-contains" | "not-contains-all" | "not-contains-any" | "not-contains-json" | "not-contains-sql" | "not-contains-xml" | "not-context-faithfulness" | "not-context-recall" | "not-context-relevance" | "not-equals" | "not-g-eval" | "not-gleu" | "not-guardrails" | "not-icontains" | "not-icontains-all" | "not-icontains-any" | "not-is-json" | "not-is-refusal" | "not-is-sql" | "not-is-valid-function-call" | "not-is-valid-openai-function-call" | "not-is-valid-openai-tools-call" | "not-is-xml" | "not-javascript" | "not-latency" | "not-levenshtein" | "not-llm-rubric" | "not-pi" | "not-meteor" | "not-model-graded-closedqa" | "not-model-graded-factuality" | "not-perplexity" | "not-perplexity-score" | "not-python" | "not-regex" | "not-rouge-n" | "not-similar" | "not-starts-with" | "not-webhook" | "select-best" | "human">;
export declare function runAssertion({ prompt, provider, assertion, test, latencyMs, providerResponse, assertIndex, }: {
    prompt?: string;
    provider?: ApiProvider;
    assertion: Assertion;
    test: AtomicTestCase;
    providerResponse: ProviderResponse;
    latencyMs?: number;
    assertIndex?: number;
}): Promise<GradingResult>;
export declare function runAssertions({ assertScoringFunction, latencyMs, prompt, provider, providerResponse, test, }: {
    assertScoringFunction?: ScoringFunction;
    latencyMs?: number;
    prompt?: string;
    provider?: ApiProvider;
    providerResponse: ProviderResponse;
    test: AtomicTestCase;
}): Promise<GradingResult>;
export declare function runCompareAssertion(test: AtomicTestCase, assertion: Assertion, outputs: string[]): Promise<GradingResult[]>;
export declare function readAssertions(filePath: string): Promise<Assertion[]>;
declare const _default: {
    runAssertion: typeof runAssertion;
    runAssertions: typeof runAssertions;
    matchesSimilarity: typeof matchesSimilarity;
    matchesClassification: typeof matchesClassification;
    matchesLlmRubric: typeof matchesLlmRubric;
    matchesFactuality: typeof matchesFactuality;
    matchesClosedQa: typeof matchesClosedQa;
    matchesAnswerRelevance: typeof matchesAnswerRelevance;
    matchesContextRecall: typeof matchesContextRecall;
    matchesContextRelevance: typeof matchesContextRelevance;
    matchesContextFaithfulness: typeof matchesContextFaithfulness;
    matchesComparisonBoolean: typeof matchesSelectBest;
    matchesModeration: typeof matchesModeration;
};
export default _default;
//# sourceMappingURL=index.d.ts.map