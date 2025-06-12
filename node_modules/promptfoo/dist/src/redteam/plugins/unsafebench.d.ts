import type { Assertion, AtomicTestCase, PluginConfig, TestCase } from '../../types';
import { RedteamPluginBase, RedteamGraderBase } from './base';
export declare const PLUGIN_ID = "promptfoo:redteam:unsafebench";
export declare const DATASET_PATH = "huggingface://datasets/yiting/UnsafeBench";
export declare const VALID_CATEGORIES: readonly ["Deception", "Harassment", "Hate", "Illegal activity", "Political", "Public and personal health", "Self-harm", "Sexual", "Shocking", "Spam", "Violence"];
export type UnsafeBenchCategory = (typeof VALID_CATEGORIES)[number];
interface UnsafeBenchPluginConfig extends PluginConfig {
    categories?: UnsafeBenchCategory[];
}
export declare class UnsafeBenchPlugin extends RedteamPluginBase {
    readonly id = "promptfoo:redteam:unsafebench";
    static readonly canGenerateRemote = false;
    private pluginConfig?;
    private datasetManager;
    constructor(provider: any, purpose: string, injectVar: string, config?: UnsafeBenchPluginConfig);
    getTemplate(): Promise<string>;
    protected getAssertions(category: string): Assertion[];
    generateTests(n: number, _delayMs?: number): Promise<TestCase[]>;
}
export declare class UnsafeBenchGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:unsafebench";
    rubric: string;
    static readonly GUIDANCE: Record<string, string>;
    getRubricForTest(test: AtomicTestCase): string;
}
export {};
//# sourceMappingURL=unsafebench.d.ts.map