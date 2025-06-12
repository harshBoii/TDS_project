import { RedteamGraderBase } from '../base';
export declare class MedicalHallucinationPluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:medical:hallucination";
    rubric: string;
    renderRubric(vars: {
        purpose: string;
        prompt: string;
        output: string;
    }): string;
}
//# sourceMappingURL=medicalHallucination.d.ts.map