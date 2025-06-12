"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DefaultSynthesizeProvider = exports.DefaultSuggestionsProvider = exports.DefaultGradingJsonProvider = exports.DefaultGradingProvider = exports.DefaultEmbeddingProvider = void 0;
const mistral_1 = require("../mistral");
exports.DefaultEmbeddingProvider = new mistral_1.MistralEmbeddingProvider();
exports.DefaultGradingProvider = new mistral_1.MistralChatCompletionProvider('mistral-large-latest');
exports.DefaultGradingJsonProvider = new mistral_1.MistralChatCompletionProvider('mistral-large-latest', {
    config: {
        response_format: { type: 'json_object' },
    },
});
exports.DefaultSuggestionsProvider = new mistral_1.MistralChatCompletionProvider('mistral-large-latest');
exports.DefaultSynthesizeProvider = new mistral_1.MistralChatCompletionProvider('mistral-large-latest');
//# sourceMappingURL=defaults.js.map