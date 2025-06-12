import { RedteamGraderBase } from '../base';
export declare class MedicalIncorrectKnowledgePluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:medical:incorrect-knowledge";
    rubric: string;
    renderRubric(vars: {
        purpose: string;
        prompt: string;
        output: string;
    }): string;
}
//# sourceMappingURL=medicalIncorrectKnowledge.d.ts.map