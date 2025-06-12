import { RedteamGraderBase } from '../base';
export declare class MedicalSycophancyPluginGrader extends RedteamGraderBase {
    readonly id = "promptfoo:redteam:medical:sycophancy";
    rubric: string;
    renderRubric(vars: {
        purpose: string;
        prompt: string;
        output: string;
    }): string;
}
//# sourceMappingURL=medicalSycophancy.d.ts.map