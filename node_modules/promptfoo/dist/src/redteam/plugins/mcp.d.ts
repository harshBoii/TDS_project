import { RedteamGraderBase } from './base';
export declare class MCPPluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:mcp";
    rubric: string;
    renderRubric(vars: {
        purpose: string;
        prompt: string;
        output: string;
    }): string;
}
//# sourceMappingURL=mcp.d.ts.map