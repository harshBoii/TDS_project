"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PromptfooModelProvider = void 0;
const cloud_1 = require("../globalConfig/cloud");
const logger_1 = __importDefault(require("../logger"));
/**
 * Provider that connects to the PromptfooModel task of the server.
 */
class PromptfooModelProvider {
    constructor(model, options = { model: '' }) {
        this.model = model || options.model;
        if (!this.model) {
            throw new Error('Model name is required for PromptfooModelProvider');
        }
        this.config = options.config || {};
        logger_1.default.debug(`[PromptfooModel] Initialized with model: ${this.model}`);
    }
    id() {
        return `promptfoo:model:${this.model}`;
    }
    async callApi(prompt, context, options) {
        logger_1.default.debug(`[PromptfooModel] Calling API with model: ${this.model}`);
        try {
            // Parse the prompt as chat messages if it's a JSON string
            let messages;
            try {
                messages = JSON.parse(prompt);
                if (!Array.isArray(messages)) {
                    messages = [{ role: 'user', content: prompt }];
                }
            }
            catch {
                // If parsing fails, assume it's a single user message
                logger_1.default.debug(`[PromptfooModel] Assuming prompt is a single user message`);
                messages = [{ role: 'user', content: prompt }];
            }
            const payload = {
                task: 'promptfoo:model',
                model: this.model,
                messages,
                config: this.config,
            };
            const baseUrl = cloud_1.cloudConfig.getApiHost();
            const url = `${baseUrl}/api/v1/task`; // Use the standard task endpoint (auth is handled conditionally on the server)
            const token = cloud_1.cloudConfig.getApiKey();
            if (!token) {
                throw new Error('No Promptfoo auth token available. Please log in with `promptfoo auth login`');
            }
            const body = JSON.stringify(payload);
            logger_1.default.debug(`[PromptfooModel] Sending request to ${url}: ${body}`);
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`,
                },
                body,
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`PromptfooModel task API error: ${response.status} ${errorText}`);
            }
            const data = await response.json();
            if (!data || !data.result) {
                throw new Error('Invalid response from PromptfooModel task API');
            }
            const modelResponse = data.result;
            logger_1.default.debug(`[PromptfooModel] Received response: ${JSON.stringify(modelResponse)}`);
            // Extract the completion from the choices
            const completionContent = modelResponse.choices?.[0]?.message?.content || '';
            // Return in the expected format for a provider
            return {
                output: completionContent,
                tokenUsage: {
                    total: modelResponse.usage?.total_tokens || 0,
                    prompt: modelResponse.usage?.prompt_tokens || 0,
                    completion: modelResponse.usage?.completion_tokens || 0,
                },
            };
        }
        catch (error) {
            logger_1.default.error(`[PromptfooModel] Error: ${error instanceof Error ? error.message : String(error)}`);
            throw error;
        }
    }
}
exports.PromptfooModelProvider = PromptfooModelProvider;
//# sourceMappingURL=promptfooModel.js.map