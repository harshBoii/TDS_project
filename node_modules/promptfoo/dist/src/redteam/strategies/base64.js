"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.addBase64Encoding = addBase64Encoding;
function addBase64Encoding(testCases, injectVar) {
    return testCases.map((testCase) => {
        const originalText = String(testCase.vars[injectVar]);
        return {
            ...testCase,
            assert: testCase.assert?.map((assertion) => ({
                ...assertion,
                metric: `${assertion.metric}/Base64`,
            })),
            vars: {
                ...testCase.vars,
                [injectVar]: Buffer.from(originalText).toString('base64'),
            },
            metadata: {
                ...testCase.metadata,
                strategyId: 'base64',
                originalText,
            },
        };
    });
}
//# sourceMappingURL=base64.js.map