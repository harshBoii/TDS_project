"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GEVAL_PROMPT_EVALUATE = exports.GEVAL_PROMPT_STEPS = void 0;
exports.readProviderPromptMap = readProviderPromptMap;
exports.processPrompt = processPrompt;
exports.readPrompts = readPrompts;
exports.processPrompts = processPrompts;
const glob_1 = require("glob");
const logger_1 = __importDefault(require("../logger"));
const util_1 = require("../util");
const fileExtensions_1 = require("../util/fileExtensions");
const invariant_1 = __importDefault(require("../util/invariant"));
const prompts_1 = require("../validators/prompts");
const csv_1 = require("./processors/csv");
const javascript_1 = require("./processors/javascript");
const jinja_1 = require("./processors/jinja");
const json_1 = require("./processors/json");
const jsonl_1 = require("./processors/jsonl");
const markdown_1 = require("./processors/markdown");
const python_1 = require("./processors/python");
const string_1 = require("./processors/string");
const text_1 = require("./processors/text");
const yaml_1 = require("./processors/yaml");
const utils_1 = require("./utils");
__exportStar(require("./grading"), exports);
/**
 * Reads and maps provider prompts based on the configuration and parsed prompts.
 * @param config - The configuration object.
 * @param parsedPrompts - Array of parsed prompts.
 * @returns A map of provider IDs to their respective prompts.
 */
function readProviderPromptMap(config, parsedPrompts) {
    const ret = {};
    if (!config.providers) {
        return ret;
    }
    const allPrompts = [];
    for (const prompt of parsedPrompts) {
        allPrompts.push(prompt.label);
    }
    if (typeof config.providers === 'string') {
        return { [config.providers]: allPrompts };
    }
    if (typeof config.providers === 'function') {
        return { 'Custom function': allPrompts };
    }
    for (const provider of config.providers) {
        if (typeof provider === 'object') {
            // It's either a ProviderOptionsMap or a ProviderOptions
            if (provider.id) {
                const rawProvider = provider;
                (0, invariant_1.default)(rawProvider.id, 'You must specify an `id` on the Provider when you override options.prompts');
                ret[rawProvider.id] = rawProvider.prompts || allPrompts;
                if (rawProvider.label) {
                    ret[rawProvider.label] = rawProvider.prompts || allPrompts;
                }
            }
            else {
                const rawProvider = provider;
                const originalId = Object.keys(rawProvider)[0];
                const providerObject = rawProvider[originalId];
                const id = providerObject.id || originalId;
                ret[id] = rawProvider[originalId].prompts || allPrompts;
            }
        }
    }
    return ret;
}
/**
 * Processes a raw prompt based on its content type and path.
 * @param prompt - The raw prompt data.
 * @param basePath - Base path for file resolution.
 * @param maxRecursionDepth - Maximum recursion depth for globbing.
 * @returns Promise resolving to an array of processed prompts.
 */
async function processPrompt(prompt, basePath = '', maxRecursionDepth = 1) {
    (0, invariant_1.default)(typeof prompt.raw === 'string', `prompt.raw must be a string, but got ${JSON.stringify(prompt.raw)}`);
    // Handling when the prompt is a raw function (e.g. javascript function)
    if (prompt.function) {
        return [prompt];
    }
    if (!(0, utils_1.maybeFilePath)(prompt.raw)) {
        return (0, string_1.processString)(prompt);
    }
    const { extension, functionName, isPathPattern, filePath, } = (0, util_1.parsePathOrGlob)(basePath, prompt.raw);
    if (isPathPattern && maxRecursionDepth > 0) {
        const globbedPath = (0, glob_1.globSync)(filePath.replace(/\\/g, '/'), {
            windowsPathsNoEscape: true,
        });
        logger_1.default.debug(`Expanded prompt ${prompt.raw} to ${filePath} and then to ${JSON.stringify(globbedPath)}`);
        const prompts = [];
        for (const globbedFilePath of globbedPath) {
            const processedPrompts = await processPrompt({ raw: globbedFilePath }, basePath, maxRecursionDepth - 1);
            prompts.push(...processedPrompts);
        }
        if (prompts.length === 0) {
            // There was nothing at this filepath, so treat it as a prompt string.
            logger_1.default.debug(`Attempted to load file at "${prompt.raw}", but no file found. Using raw string.`);
            prompts.push(...(0, string_1.processString)(prompt));
        }
        return prompts;
    }
    if (extension === '.csv') {
        return (0, csv_1.processCsvPrompts)(filePath, prompt);
    }
    if (extension === '.j2') {
        return (0, jinja_1.processJinjaFile)(filePath, prompt);
    }
    if (extension === '.json') {
        return (0, json_1.processJsonFile)(filePath, prompt);
    }
    if (extension === '.jsonl') {
        return (0, jsonl_1.processJsonlFile)(filePath, prompt);
    }
    if (extension && (0, fileExtensions_1.isJavascriptFile)(extension)) {
        return (0, javascript_1.processJsFile)(filePath, prompt, functionName);
    }
    if (extension === '.md') {
        return (0, markdown_1.processMarkdownFile)(filePath, prompt);
    }
    if (extension === '.py') {
        return (0, python_1.processPythonFile)(filePath, prompt, functionName);
    }
    if (extension === '.txt') {
        return (0, text_1.processTxtFile)(filePath, prompt);
    }
    if (extension && ['.yml', '.yaml'].includes(extension)) {
        return (0, yaml_1.processYamlFile)(filePath, prompt);
    }
    return [];
}
/**
 * Reads and processes prompts from a specified path or glob pattern.
 * @param promptPathOrGlobs - The path or glob pattern.
 * @param basePath - Base path for file resolution.
 * @returns Promise resolving to an array of processed prompts.
 */
async function readPrompts(promptPathOrGlobs, basePath = '') {
    logger_1.default.debug(`Reading prompts from ${JSON.stringify(promptPathOrGlobs)}`);
    const promptPartials = (0, utils_1.normalizeInput)(promptPathOrGlobs);
    const prompts = [];
    for (const prompt of promptPartials) {
        const promptBatch = await processPrompt(prompt, basePath);
        if (promptBatch.length === 0) {
            throw new Error(`There are no prompts in ${JSON.stringify(prompt.raw)}`);
        }
        prompts.push(...promptBatch);
    }
    return prompts;
}
async function processPrompts(prompts) {
    return (await Promise.all(prompts.map(async (promptInput) => {
        if (typeof promptInput === 'function') {
            return {
                raw: promptInput.toString(),
                label: promptInput?.name ?? promptInput.toString(),
                function: promptInput,
            };
        }
        else if (typeof promptInput === 'string') {
            return readPrompts(promptInput);
        }
        try {
            return prompts_1.PromptSchema.parse(promptInput);
        }
        catch (error) {
            logger_1.default.warn(`Prompt input is not a valid prompt schema: ${error}\nFalling back to serialized JSON as raw prompt.`);
            return {
                raw: JSON.stringify(promptInput),
                label: JSON.stringify(promptInput),
            };
        }
    }))).flat();
}
exports.GEVAL_PROMPT_STEPS = `
Given an evaluation criteria which outlines how you should judge some text, generate 3-4 concise evaluation steps for any text based on the criteria below.

Evaluation Criteria:
{{criteria}}

**
IMPORTANT: Please make sure to only return in minified JSON format, with the "steps" key as a list of strings. No additional words, explanation or formatting is needed.
Example JSON:
{"steps": <list_of_strings>}
**

JSON:
`;
exports.GEVAL_PROMPT_EVALUATE = `
You will be given one Reply for a Source Text below. Your task is to rate the Reply on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
{{criteria}}

Evaluation Steps:
- {{steps}}
- Given the evaluation steps, return a JSON with two keys: 1) a "score" key ranging from 0 - {{maxScore}}, with {{maxScore}} being that it follows the Evaluation Criteria outlined in the Evaluation Steps and 0 being that it does not; 2) a "reason" key, a reason for the given score, but DO NOT QUOTE THE SCORE in your reason. Please mention specific information from Source Text and Reply in your reason, but be very concise with it!

Source Text:
{{input}}

Reply:
{{output}}

**
IMPORTANT: Please make sure to only return in minified JSON format, with the "score" and "reason" key. No additional words, explanation or formatting is needed.

Example JSON:
{"score":0,"reason":"The text does not follow the evaluation steps provided."}
**

JSON:
`;
//# sourceMappingURL=index.js.map