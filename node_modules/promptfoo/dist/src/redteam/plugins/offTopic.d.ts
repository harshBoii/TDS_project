import { RedteamGraderBase } from './base';
export declare class OffTopicPluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:off-topic";
    rubric: string;
    renderRubric(vars: {
        purpose: string;
        prompt: string;
        output: string;
    }): string;
}
//# sourceMappingURL=offTopic.d.ts.map