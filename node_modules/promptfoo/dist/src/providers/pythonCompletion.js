"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PythonProvider = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const cache_1 = require("../cache");
const logger_1 = __importDefault(require("../logger"));
const pythonUtils_1 = require("../python/pythonUtils");
const util_1 = require("../util");
const createHash_1 = require("../util/createHash");
const fileReference_1 = require("../util/fileReference");
const json_1 = require("../util/json");
class PythonProvider {
    constructor(runPath, options) {
        this.options = options;
        this.isInitialized = false;
        this.initializationPromise = null;
        const { filePath: providerPath, functionName } = (0, util_1.parsePathOrGlob)(options?.config.basePath || '', runPath);
        this.scriptPath = path_1.default.relative(options?.config.basePath || '', providerPath);
        this.functionName = functionName || null;
        this.id = () => options?.id ?? `python:${this.scriptPath}:${this.functionName || 'default'}`;
        this.label = options?.label;
        this.config = options?.config ?? {};
    }
    id() {
        return `python:${this.scriptPath}:${this.functionName || 'default'}`;
    }
    /**
     * Process any file:// references in the configuration
     * This should be called after initialization
     * @returns A promise that resolves when all file references have been processed
     */
    async initialize() {
        // If already initialized, return immediately
        if (this.isInitialized) {
            return;
        }
        // If initialization is in progress, return the existing promise
        if (this.initializationPromise) {
            return this.initializationPromise;
        }
        // Start initialization and store the promise
        this.initializationPromise = (async () => {
            try {
                this.config = await (0, fileReference_1.processConfigFileReferences)(this.config, this.options?.config.basePath || '');
                this.isInitialized = true;
                logger_1.default.debug(`Initialized Python provider ${this.id()}`);
            }
            catch (error) {
                // Reset the initialization promise so future calls can retry
                this.initializationPromise = null;
                throw error;
            }
        })();
        return this.initializationPromise;
    }
    /**
     * Execute the Python script with the specified API type
     * Handles caching, file reference processing, and executing the Python script
     *
     * @param prompt - The prompt to pass to the Python script
     * @param context - Optional context information
     * @param apiType - The type of API to call (call_api, call_embedding_api, call_classification_api)
     * @returns The response from the Python script
     */
    async executePythonScript(prompt, context, apiType) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        const absPath = path_1.default.resolve(path_1.default.join(this.options?.config.basePath || '', this.scriptPath));
        logger_1.default.debug(`Computing file hash for script ${absPath}`);
        const fileHash = (0, createHash_1.sha256)(fs_1.default.readFileSync(absPath, 'utf-8'));
        // Create cache key including the function name to ensure different functions don't share caches
        const cacheKey = `python:${this.scriptPath}:${this.functionName || 'default'}:${apiType}:${fileHash}:${prompt}:${JSON.stringify(this.options)}:${JSON.stringify(context?.vars)}`;
        logger_1.default.debug(`PythonProvider cache key: ${cacheKey}`);
        const cache = await (0, cache_1.getCache)();
        let cachedResult;
        const cacheEnabled = (0, cache_1.isCacheEnabled)();
        logger_1.default.debug(`PythonProvider cache enabled: ${cacheEnabled}`);
        if (cacheEnabled) {
            cachedResult = await cache.get(cacheKey);
            logger_1.default.debug(`PythonProvider cache hit: ${Boolean(cachedResult)}`);
        }
        if (cachedResult) {
            logger_1.default.debug(`Returning cached ${apiType} result for script ${absPath}`);
            const parsedResult = JSON.parse(cachedResult);
            logger_1.default.debug(`PythonProvider parsed cached result type: ${typeof parsedResult}, keys: ${Object.keys(parsedResult).join(',')}`);
            // IMPORTANT: Set cached flag to true so evaluator recognizes this as cached
            if (apiType === 'call_api' && typeof parsedResult === 'object' && parsedResult !== null) {
                logger_1.default.debug(`PythonProvider setting cached=true for cached ${apiType} result`);
                parsedResult.cached = true;
                // Update token usage format for cached results
                if (parsedResult.tokenUsage) {
                    const total = parsedResult.tokenUsage.total || 0;
                    parsedResult.tokenUsage = {
                        cached: total,
                        total,
                    };
                    logger_1.default.debug(`Updated token usage for cached result: ${JSON.stringify(parsedResult.tokenUsage)}`);
                }
            }
            return parsedResult;
        }
        else {
            if (context) {
                // Remove properties not useful in Python
                delete context.fetchWithCache;
                delete context.getCache;
                delete context.logger;
            }
            // Create a new options object with processed file references included in the config
            // This ensures any file:// references are replaced with their actual content
            const optionsWithProcessedConfig = {
                ...this.options,
                config: {
                    ...this.options?.config,
                    ...this.config, // Merge in the processed config containing resolved file references
                },
            };
            // Prepare arguments for the Python script based on API type
            const args = apiType === 'call_api'
                ? [prompt, optionsWithProcessedConfig, context]
                : [prompt, optionsWithProcessedConfig];
            logger_1.default.debug(`Running python script ${absPath} with scriptPath ${this.scriptPath} and args: ${(0, json_1.safeJsonStringify)(args)}`);
            const functionName = this.functionName || apiType;
            let result;
            switch (apiType) {
                case 'call_api':
                    result = await (0, pythonUtils_1.runPython)(absPath, functionName, args, {
                        pythonExecutable: this.config.pythonExecutable,
                    });
                    // Log result structure for debugging
                    logger_1.default.debug(`Python provider result structure: ${result ? typeof result : 'undefined'}, keys: ${result ? Object.keys(result).join(',') : 'none'}`);
                    if (result && 'output' in result) {
                        logger_1.default.debug(`Python provider output type: ${typeof result.output}, isArray: ${Array.isArray(result.output)}`);
                    }
                    if (!result ||
                        typeof result !== 'object' ||
                        (!('output' in result) && !('error' in result))) {
                        throw new Error(`The Python script \`${functionName}\` function must return a dict with an \`output\` string/object or \`error\` string, instead got: ${JSON.stringify(result)}`);
                    }
                    break;
                case 'call_embedding_api':
                    result = await (0, pythonUtils_1.runPython)(absPath, functionName, args, {
                        pythonExecutable: this.config.pythonExecutable,
                    });
                    if (!result ||
                        typeof result !== 'object' ||
                        (!('embedding' in result) && !('error' in result))) {
                        throw new Error(`The Python script \`${functionName}\` function must return a dict with an \`embedding\` array or \`error\` string, instead got ${JSON.stringify(result)}`);
                    }
                    break;
                case 'call_classification_api':
                    result = await (0, pythonUtils_1.runPython)(absPath, functionName, args, {
                        pythonExecutable: this.config.pythonExecutable,
                    });
                    if (!result ||
                        typeof result !== 'object' ||
                        (!('classification' in result) && !('error' in result))) {
                        throw new Error(`The Python script \`${functionName}\` function must return a dict with a \`classification\` object or \`error\` string, instead of ${JSON.stringify(result)}`);
                    }
                    break;
                default:
                    throw new Error(`Unsupported apiType: ${apiType}`);
            }
            // Store result in cache if enabled and no errors
            const hasError = 'error' in result &&
                result.error !== null &&
                result.error !== undefined &&
                result.error !== '';
            if ((0, cache_1.isCacheEnabled)() && !hasError) {
                logger_1.default.debug(`PythonProvider caching result: ${cacheKey}`);
                await cache.set(cacheKey, JSON.stringify(result));
            }
            else {
                logger_1.default.debug(`PythonProvider not caching result: ${(0, cache_1.isCacheEnabled)() ? (hasError ? 'has error' : 'unknown reason') : 'cache disabled'}`);
            }
            // Set cached=false on fresh results
            if (typeof result === 'object' && result !== null && apiType === 'call_api') {
                logger_1.default.debug(`PythonProvider explicitly setting cached=false for fresh result`);
                result.cached = false;
            }
            return result;
        }
    }
    async callApi(prompt, context) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        return this.executePythonScript(prompt, context, 'call_api');
    }
    async callEmbeddingApi(prompt) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        return this.executePythonScript(prompt, undefined, 'call_embedding_api');
    }
    async callClassificationApi(prompt) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        return this.executePythonScript(prompt, undefined, 'call_classification_api');
    }
}
exports.PythonProvider = PythonProvider;
//# sourceMappingURL=pythonCompletion.js.map