import { RedteamGraderBase } from '../base';
export declare class MedicalPrioritizationErrorPluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:medical:prioritization-error";
    rubric: string;
    renderRubric(vars: {
        purpose: string;
        prompt: string;
        output: string;
    }): string;
}
//# sourceMappingURL=medicalPrioritizationError.d.ts.map