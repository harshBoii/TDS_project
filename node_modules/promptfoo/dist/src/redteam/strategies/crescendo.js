"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.addCrescendo = addCrescendo;
function addCrescendo(testCases, injectVar, config) {
    return testCases.map((testCase) => {
        const originalText = String(testCase.vars[injectVar]);
        return {
            ...testCase,
            provider: {
                id: 'promptfoo:redteam:crescendo',
                config: {
                    injectVar,
                    ...config,
                },
            },
            assert: testCase.assert?.map((assertion) => ({
                ...assertion,
                metric: `${assertion.metric}/Crescendo`,
            })),
            metadata: {
                ...testCase.metadata,
                strategyId: 'crescendo',
                originalText,
            },
        };
    });
}
//# sourceMappingURL=crescendo.js.map