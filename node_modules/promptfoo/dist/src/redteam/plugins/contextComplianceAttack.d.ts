import type { AssertionValue, ResultSuggestion } from '../../types';
import { RedteamGraderBase } from './base';
export declare const PLUGIN_ID = "promptfoo:redteam:cca";
export declare class CcaGrader extends RedteamGraderBase {
    id: string;
    rubric: string;
    getSuggestions({ rawPrompt, renderedValue, }: {
        rawPrompt: string;
        renderedValue?: AssertionValue;
    }): ResultSuggestion[];
}
//# sourceMappingURL=contextComplianceAttack.d.ts.map