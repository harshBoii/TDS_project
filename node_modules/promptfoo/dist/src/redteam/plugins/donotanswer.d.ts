import type { Assertion, TestCase } from '../../types';
import { RedteamPluginBase } from './base';
export declare const PLUGIN_ID = "promptfoo:redteam:donotanswer";
type DoNotAnswerVars = Record<string, string>;
interface DoNotAnswerTestCase extends TestCase {
    vars: DoNotAnswerVars;
}
export declare function fetchDataset(limit: number): Promise<DoNotAnswerTestCase[]>;
export declare class DoNotAnswerPlugin extends RedteamPluginBase {
    readonly id = "promptfoo:redteam:donotanswer";
    static readonly canGenerateRemote = false;
    getTemplate(): Promise<string>;
    getAssertions(prompt: string): Assertion[];
    generateTests(n: number, _delayMs?: number): Promise<TestCase[]>;
}
export {};
//# sourceMappingURL=donotanswer.d.ts.map