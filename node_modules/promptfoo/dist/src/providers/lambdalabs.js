"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createLambdaLabsProvider = createLambdaLabsProvider;
const chat_1 = require("./openai/chat");
const completion_1 = require("./openai/completion");
const embedding_1 = require("./openai/embedding");
/**
 * Creates a Lambda Labs provider using OpenAI-compatible endpoints
 *
 * Documentation: https://docs.lambdalabs.com/api
 *
 * Lambda Labs API supports the OpenAI API format and can be used as a drop-in replacement.
 * All parameters are automatically passed through to the Lambda Labs API.
 */
function createLambdaLabsProvider(providerPath, options = {}) {
    const splits = providerPath.split(':');
    const config = options.config?.config || {};
    const lambdaLabsConfig = {
        ...options,
        config: {
            apiBaseUrl: 'https://api.lambda.ai/v1',
            apiKeyEnvar: 'LAMBDA_API_KEY',
            passthrough: {
                ...config,
            },
        },
    };
    if (splits[1] === 'chat') {
        const modelName = splits.slice(2).join(':');
        return new chat_1.OpenAiChatCompletionProvider(modelName, lambdaLabsConfig);
    }
    else if (splits[1] === 'completion') {
        const modelName = splits.slice(2).join(':');
        return new completion_1.OpenAiCompletionProvider(modelName, lambdaLabsConfig);
    }
    else if (splits[1] === 'embedding' || splits[1] === 'embeddings') {
        const modelName = splits.slice(2).join(':');
        return new embedding_1.OpenAiEmbeddingProvider(modelName, lambdaLabsConfig);
    }
    else {
        // If no specific type is provided, default to chat
        const modelName = splits.slice(1).join(':');
        return new chat_1.OpenAiChatCompletionProvider(modelName, lambdaLabsConfig);
    }
}
//# sourceMappingURL=lambdalabs.js.map