import type { Assertion, TestCase } from '../../types';
import { RedteamPluginBase } from './base';
export declare const PLUGIN_ID = "promptfoo:redteam:xstest";
type XSTestVars = Record<string, string>;
interface XSTestTestCase extends TestCase {
    vars: XSTestVars;
}
export declare function fetchDataset(limit: number): Promise<XSTestTestCase[]>;
export declare class XSTestPlugin extends RedteamPluginBase {
    readonly id = "promptfoo:redteam:xstest";
    static readonly canGenerateRemote = false;
    getTemplate(): Promise<string>;
    getAssertions(prompt: string): Assertion[];
    generateTests(n: number, _delayMs?: number): Promise<TestCase[]>;
}
export {};
//# sourceMappingURL=xstest.d.ts.map