import z from 'zod';
import { RedteamGraderBase } from '../base';
export declare const VarsSchema: z.ZodObject<{
    purpose: z.ZodString;
    prompt: z.ZodString;
    scenario: z.ZodObject<{
        memory: z.ZodString;
        followUp: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        memory: string;
        followUp: string;
    }, {
        memory: string;
        followUp: string;
    }>;
}, "strip", z.ZodTypeAny, {
    prompt: string;
    purpose: string;
    scenario: {
        memory: string;
        followUp: string;
    };
}, {
    prompt: string;
    purpose: string;
    scenario: {
        memory: string;
        followUp: string;
    };
}>;
type Vars = z.infer<typeof VarsSchema>;
export declare class MemoryPoisoningPluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:agentic:memory-poisoning";
    rubric: string;
    renderRubric(vars: Vars): string;
}
export {};
//# sourceMappingURL=memoryPoisoning.d.ts.map