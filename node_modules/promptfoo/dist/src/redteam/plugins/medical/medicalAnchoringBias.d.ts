import { RedteamGraderBase } from '../base';
export declare class MedicalAnchoringBiasPluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:medical:anchoring-bias";
    rubric: string;
    renderRubric(vars: {
        purpose: string;
        prompt: string;
        output: string;
    }): string;
}
//# sourceMappingURL=medicalAnchoringBias.d.ts.map