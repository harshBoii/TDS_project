export interface GuardResult {
    model: string;
    results: Array<{
        categories: Record<string, boolean>;
        category_scores: Record<string, number>;
        flagged: boolean;
        payload?: {
            pii?: Array<{
                entity_type: string;
                start: number;
                end: number;
                pii: string;
            }>;
        };
    }>;
}
export interface AdaptiveRequest {
    prompt: string;
    policies?: string[];
}
export interface AdaptiveResult {
    model: string;
    adaptedPrompt: string;
    modifications: Array<{
        type: string;
        reason: string;
        original: string;
        modified: string;
    }>;
}
declare const guardrails: {
    guard(input: string): Promise<GuardResult>;
    pii(input: string): Promise<GuardResult>;
    harm(input: string): Promise<GuardResult>;
    adaptive(request: AdaptiveRequest): Promise<AdaptiveResult>;
};
export default guardrails;
//# sourceMappingURL=guardrails.d.ts.map