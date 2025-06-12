"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.loadFileReference = loadFileReference;
exports.processConfigFileReferences = processConfigFileReferences;
const fs_1 = __importDefault(require("fs"));
const js_yaml_1 = __importDefault(require("js-yaml"));
const path_1 = __importDefault(require("path"));
const esm_1 = require("../esm");
const logger_1 = __importDefault(require("../logger"));
const pythonUtils_1 = require("../python/pythonUtils");
const fileExtensions_1 = require("./fileExtensions");
/**
 * Loads the content from a file reference
 * @param fileRef The file reference string (e.g. 'file://path/to/file.json')
 * @param basePath Base path for resolving relative paths
 * @returns The loaded content from the file
 */
async function loadFileReference(fileRef, basePath = '') {
    // Remove the file:// prefix
    const pathWithProtocolRemoved = fileRef.slice('file://'.length);
    // Split to check for function name
    const parts = pathWithProtocolRemoved.split(':');
    const filePath = parts[0];
    const functionName = parts.length > 1 ? parts[1] : undefined;
    // Resolve the absolute path
    const resolvedPath = path_1.default.resolve(basePath, filePath);
    const extension = path_1.default.extname(resolvedPath).toLowerCase();
    logger_1.default.debug(`Loading file reference: ${fileRef}, resolvedPath: ${resolvedPath}, extension: ${extension}`);
    try {
        if (extension === '.json') {
            logger_1.default.debug(`Loading JSON file: ${resolvedPath}`);
            const content = await fs_1.default.promises.readFile(resolvedPath, 'utf8');
            return JSON.parse(content);
        }
        else if (extension === '.yaml' || extension === '.yml') {
            logger_1.default.debug(`Loading YAML file: ${resolvedPath}`);
            const content = await fs_1.default.promises.readFile(resolvedPath, 'utf8');
            return js_yaml_1.default.load(content);
        }
        else if ((0, fileExtensions_1.isJavascriptFile)(resolvedPath)) {
            logger_1.default.debug(`Loading JavaScript file: ${resolvedPath}`);
            const mod = await (0, esm_1.importModule)(resolvedPath, functionName);
            return typeof mod === 'function' ? await mod() : mod;
        }
        else if (extension === '.py') {
            logger_1.default.debug(`Loading Python file: ${resolvedPath}, function: ${functionName || 'get_config'}`);
            const fnName = functionName || 'get_config';
            const result = await (0, pythonUtils_1.runPython)(resolvedPath, fnName, []);
            return result;
        }
        else if (extension === '.txt' || extension === '.md' || extension === '') {
            // For text files, just return the content as a string
            logger_1.default.debug(`Loading text file: ${resolvedPath}`);
            return await fs_1.default.promises.readFile(resolvedPath, 'utf8');
        }
        else {
            logger_1.default.debug(`Unsupported file extension: ${extension}`);
            throw new Error(`Unsupported file extension: ${extension}`);
        }
    }
    catch (error) {
        logger_1.default.error(`Error loading file reference ${fileRef}: ${error}`);
        throw error;
    }
}
/**
 * Recursively processes a configuration object, replacing any file:// references
 * with the content of the referenced files
 * @param config The configuration object to process
 * @param basePath Base path for resolving relative paths
 * @returns A new configuration object with file references resolved
 */
async function processConfigFileReferences(config, basePath = '') {
    if (!config) {
        return config;
    }
    // Handle string values with file:// protocol
    if (typeof config === 'string' && config.startsWith('file://')) {
        return await loadFileReference(config, basePath);
    }
    // Handle arrays
    if (Array.isArray(config)) {
        const result = [];
        for (const item of config) {
            result.push(await processConfigFileReferences(item, basePath));
        }
        return result;
    }
    // Handle objects
    if (typeof config === 'object' && config !== null) {
        const result = {};
        for (const [key, value] of Object.entries(config)) {
            result[key] = await processConfigFileReferences(value, basePath);
        }
        return result;
    }
    // Return primitive values as is
    return config;
}
//# sourceMappingURL=fileReference.js.map