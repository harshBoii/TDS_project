"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.handleIsValidFunctionCall = void 0;
const ai_studio_1 = require("../providers/google/ai.studio");
const live_1 = require("../providers/google/live");
const util_1 = require("../providers/google/util");
const vertex_1 = require("../providers/google/vertex");
const chat_1 = require("../providers/openai/chat");
const util_2 = require("../providers/openai/util");
const handleIsValidFunctionCall = ({ assertion, output, provider, test, }) => {
    try {
        if (provider instanceof ai_studio_1.AIStudioChatProvider ||
            provider instanceof live_1.GoogleLiveProvider ||
            provider instanceof vertex_1.VertexChatProvider) {
            (0, util_1.validateFunctionCall)(output, provider.config?.tools, test.vars);
        }
        else if (provider instanceof chat_1.OpenAiChatCompletionProvider) {
            (0, util_2.validateFunctionCall)(output, provider.config.functions, test.vars);
        }
        else {
            throw new Error(`Provider does not have functionality for checking function call.`);
        }
        return {
            pass: true,
            score: 1,
            reason: 'Assertion passed',
            assertion,
        };
    }
    catch (err) {
        return {
            pass: false,
            score: 0,
            reason: err.message,
            assertion,
        };
    }
};
exports.handleIsValidFunctionCall = handleIsValidFunctionCall;
//# sourceMappingURL=functionToolCall.js.map