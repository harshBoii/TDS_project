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
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.maybeCoerceToGeminiFormat = maybeCoerceToGeminiFormat;
exports.getGoogleClient = getGoogleClient;
exports.hasGoogleDefaultCredentials = hasGoogleDefaultCredentials;
exports.getCandidate = getCandidate;
exports.formatCandidateContents = formatCandidateContents;
exports.mergeParts = mergeParts;
exports.normalizeTools = normalizeTools;
exports.loadFile = loadFile;
exports.geminiFormatAndSystemInstructions = geminiFormatAndSystemInstructions;
exports.parseStringObject = parseStringObject;
exports.validateFunctionCall = validateFunctionCall;
const rfdc_1 = __importDefault(require("rfdc"));
const zod_1 = require("zod");
const logger_1 = __importDefault(require("../../logger"));
const util_1 = require("../../util");
const file_1 = require("../../util/file");
const json_1 = require("../../util/json");
const templates_1 = require("../../util/templates");
const shared_1 = require("../shared");
const types_1 = require("./types");
const ajv = (0, json_1.getAjv)();
// property_ordering is an optional field sometimes present in gemini tool configs, but ajv doesn't know about it.
// At the moment we will just ignore it, so the is-valid-function-call won't check property field ordering.
ajv.addKeyword('property_ordering');
const clone = (0, rfdc_1.default)();
const PartSchema = zod_1.z.object({
    text: zod_1.z.string().optional(),
    inline_data: zod_1.z
        .object({
        mime_type: zod_1.z.string(),
        data: zod_1.z.string(),
    })
        .optional(),
});
const ContentSchema = zod_1.z.object({
    role: zod_1.z.enum(['user', 'model']).optional(),
    parts: zod_1.z.array(PartSchema),
});
const GeminiFormatSchema = zod_1.z.array(ContentSchema);
function maybeCoerceToGeminiFormat(contents) {
    let coerced = false;
    const parseResult = GeminiFormatSchema.safeParse(contents);
    if (parseResult.success) {
        // Check for native Gemini system_instruction format
        let systemInst = undefined;
        if (typeof contents === 'object' && 'system_instruction' in contents) {
            systemInst = contents.system_instruction;
            // We need to modify the contents to remove system_instruction
            // since it's already extracted to systemInst
            if (typeof contents === 'object' && 'contents' in contents) {
                contents = contents.contents;
            }
            coerced = true;
        }
        return {
            contents: parseResult.data,
            coerced,
            systemInstruction: systemInst,
        };
    }
    let coercedContents;
    // Handle native Gemini format with system_instruction
    if (typeof contents === 'object' &&
        !Array.isArray(contents) &&
        'system_instruction' in contents) {
        const systemInst = contents.system_instruction;
        if ('contents' in contents) {
            coercedContents = contents.contents;
        }
        else {
            // If contents field is not present, use an empty array
            coercedContents = [];
        }
        return {
            contents: coercedContents,
            coerced: true,
            systemInstruction: systemInst,
        };
    }
    if (typeof contents === 'string') {
        coercedContents = [
            {
                parts: [{ text: contents }],
            },
        ];
        coerced = true;
    }
    else if (Array.isArray(contents) &&
        contents.every((item) => typeof item.content === 'string')) {
        // This looks like an OpenAI chat format
        coercedContents = contents.map((item) => ({
            role: item.role,
            parts: [{ text: item.content }],
        }));
        coerced = true;
    }
    else if (Array.isArray(contents) && contents.every((item) => item.role && item.content)) {
        // This looks like an OpenAI chat format with content that might be an array or object
        coercedContents = contents.map((item) => {
            if (Array.isArray(item.content)) {
                // Handle array content
                const parts = item.content.map((contentItem) => {
                    if (typeof contentItem === 'string') {
                        return { text: contentItem };
                    }
                    else if (contentItem.type === 'text') {
                        return { text: contentItem.text };
                    }
                    else {
                        // Handle other content types if needed
                        return contentItem;
                    }
                });
                return {
                    role: item.role,
                    parts,
                };
            }
            else if (typeof item.content === 'object') {
                // Handle object content
                return {
                    role: item.role,
                    parts: [item.content],
                };
            }
            else {
                // Handle string content
                return {
                    role: item.role,
                    parts: [{ text: item.content }],
                };
            }
        });
        coerced = true;
    }
    else if (typeof contents === 'object' && 'parts' in contents) {
        // This might be a single content object
        coercedContents = [contents];
        coerced = true;
    }
    else {
        logger_1.default.warn(`Unknown format for Gemini: ${JSON.stringify(contents)}`);
        return { contents: contents, coerced: false, systemInstruction: undefined };
    }
    const systemPromptParts = [];
    coercedContents = coercedContents.filter((message) => {
        if (message.role === 'system' && message.parts.length > 0) {
            systemPromptParts.push(...message.parts.filter((part) => 'text' in part && typeof part.text === 'string'));
            return false;
        }
        return true;
    });
    return {
        contents: coercedContents,
        coerced,
        systemInstruction: systemPromptParts.length > 0 ? { parts: systemPromptParts } : undefined,
    };
}
let cachedAuth;
async function getGoogleClient() {
    if (!cachedAuth) {
        let GoogleAuth;
        try {
            const importedModule = await Promise.resolve().then(() => __importStar(require('google-auth-library')));
            GoogleAuth = importedModule.GoogleAuth;
        }
        catch {
            throw new Error('The google-auth-library package is required as a peer dependency. Please install it in your project or globally.');
        }
        cachedAuth = new GoogleAuth({
            scopes: 'https://www.googleapis.com/auth/cloud-platform',
        });
    }
    const client = await cachedAuth.getClient();
    const projectId = await cachedAuth.getProjectId();
    return { client, projectId };
}
async function hasGoogleDefaultCredentials() {
    try {
        await getGoogleClient();
        return true;
    }
    catch {
        return false;
    }
}
function getCandidate(data) {
    if (!(data && data.candidates && data.candidates.length === 1)) {
        throw new Error('Expected one candidate in API response.');
    }
    const candidate = data.candidates[0];
    return candidate;
}
function formatCandidateContents(candidate) {
    if (candidate.content?.parts) {
        let output = '';
        let is_text = true;
        for (const part of candidate.content.parts) {
            if ('text' in part) {
                output += part.text;
            }
            else {
                is_text = false;
            }
        }
        if (is_text) {
            return output;
        }
        else {
            return candidate.content.parts;
        }
    }
    else {
        throw new Error(`No output found in response: ${JSON.stringify(candidate)}`);
    }
}
function mergeParts(parts1, parts2) {
    if (parts1 === undefined) {
        return parts2;
    }
    if (typeof parts1 === 'string' && typeof parts2 === 'string') {
        return parts1 + parts2;
    }
    const array1 = typeof parts1 === 'string' ? [{ text: parts1 }] : parts1;
    const array2 = typeof parts2 === 'string' ? [{ text: parts2 }] : parts2;
    array1.push(...array2);
    return array1;
}
/**
 * Normalizes tools configuration to handle both snake_case and camelCase formats.
 * This ensures compatibility with both Google API formats while maintaining
 * consistent behavior in our codebase.
 */
function normalizeTools(tools) {
    return tools.map((tool) => {
        const normalizedTool = { ...tool };
        // Use index access with type assertion to avoid TypeScript errors
        // Handle google_search -> googleSearch conversion
        if (tool.google_search && !normalizedTool.googleSearch) {
            normalizedTool.googleSearch = tool.google_search;
        }
        // Handle code_execution -> codeExecution conversion
        if (tool.code_execution && !normalizedTool.codeExecution) {
            normalizedTool.codeExecution = tool.code_execution;
        }
        // Handle google_search_retrieval -> googleSearchRetrieval conversion
        if (tool.google_search_retrieval && !normalizedTool.googleSearchRetrieval) {
            normalizedTool.googleSearchRetrieval = tool.google_search_retrieval;
        }
        return normalizedTool;
    });
}
function loadFile(config_var, context_vars) {
    // Ensures that files are loaded correctly. Files may be defined in multiple ways:
    // 1. Directly in the provider:
    //    config_var will be the file path, which will be loaded here in maybeLoadFromExternalFile.
    // 2. In a test variable that is used in the provider via a nunjucks:
    //    context_vars will contain a string of the contents of the file with whitespace.
    //    This will be inserted into the nunjucks in contfig_tools and the output needs to be parsed.
    const fileContents = (0, file_1.maybeLoadFromExternalFile)((0, util_1.renderVarsInObject)(config_var, context_vars));
    if (typeof fileContents === 'string') {
        try {
            const parsedContents = JSON.parse(fileContents);
            return Array.isArray(parsedContents) ? normalizeTools(parsedContents) : parsedContents;
        }
        catch (err) {
            logger_1.default.debug(`ERROR: failed to convert file contents to JSON:\n${JSON.stringify(err)}`);
            return fileContents;
        }
    }
    // If fileContents is already an array of tools, normalize them
    if (Array.isArray(fileContents)) {
        return normalizeTools(fileContents);
    }
    return fileContents;
}
function geminiFormatAndSystemInstructions(prompt, contextVars, configSystemInstruction) {
    let contents = (0, shared_1.parseChatPrompt)(prompt, [
        {
            parts: [
                {
                    text: prompt,
                },
            ],
            role: 'user',
        },
    ]);
    const { contents: updatedContents, coerced, systemInstruction: parsedSystemInstruction, } = maybeCoerceToGeminiFormat(contents);
    if (coerced) {
        logger_1.default.debug(`Coerced JSON prompt to Gemini format: ${JSON.stringify(contents)}`);
        contents = updatedContents;
    }
    let systemInstruction = parsedSystemInstruction;
    if (configSystemInstruction && !systemInstruction) {
        // Make a copy
        systemInstruction = clone(configSystemInstruction);
        // Load SI from file
        if (typeof configSystemInstruction === 'string') {
            systemInstruction = loadFile(configSystemInstruction, contextVars);
        }
        // Format SI if string was not a filepath above
        if (typeof systemInstruction === 'string') {
            systemInstruction = { parts: [{ text: systemInstruction }] };
        }
        if (contextVars && systemInstruction) {
            const nunjucks = (0, templates_1.getNunjucksEngine)();
            for (const part of systemInstruction.parts) {
                if (part.text) {
                    try {
                        part.text = nunjucks.renderString(part.text, contextVars);
                    }
                    catch (err) {
                        throw new Error(`Unable to render nunjunks in systemInstruction: ${err}`);
                    }
                }
            }
        }
    }
    else if (configSystemInstruction && systemInstruction) {
        throw new Error(`Template error: system instruction defined in prompt and config.`);
    }
    return { contents, systemInstruction };
}
/**
 * Recursively traverses a JSON schema object and converts
 * uppercase type keywords (string values) to lowercase.
 * Handles nested objects and arrays within the schema.
 * Creates a deep copy to avoid modifying the original schema.
 *
 * @param {object | any} schemaNode - The current node (object or value) being processed.
 * @returns {object | any} - The processed node with type keywords lowercased.
 */
function normalizeSchemaTypes(schemaNode) {
    // Handle non-objects (including null) and arrays directly by iterating/returning
    if (typeof schemaNode !== 'object' || schemaNode === null) {
        return schemaNode;
    }
    if (Array.isArray(schemaNode)) {
        return schemaNode.map(normalizeSchemaTypes); // Recurse for array elements
    }
    // Create a new object to avoid modifying the original
    const newNode = {};
    for (const key in schemaNode) {
        if (Object.prototype.hasOwnProperty.call(schemaNode, key)) {
            const value = schemaNode[key];
            if (key === 'type') {
                if (typeof value === 'string' &&
                    types_1.VALID_SCHEMA_TYPES.includes(value)) {
                    // Convert type value(s) to lowercase
                    newNode[key] = value.toLowerCase();
                }
                else if (Array.isArray(value)) {
                    // Handle type arrays like ["STRING", "NULL"]
                    newNode[key] = value.map((t) => typeof t === 'string' && types_1.VALID_SCHEMA_TYPES.includes(t)
                        ? t.toLowerCase()
                        : t);
                }
                else {
                    // Handle type used as function field rather than a schema type definition
                    newNode[key] = normalizeSchemaTypes(value);
                }
            }
            else {
                // Recursively process nested objects/arrays
                newNode[key] = normalizeSchemaTypes(value);
            }
        }
    }
    return newNode;
}
function parseStringObject(input) {
    if (typeof input === 'string') {
        return JSON.parse(input);
    }
    return input;
}
function validateFunctionCall(output, functions, vars) {
    let functionCalls;
    try {
        let parsedOutput = parseStringObject(output);
        if ('toolCall' in parsedOutput) {
            // Live Format
            parsedOutput = parsedOutput.toolCall;
            functionCalls = parsedOutput.functionCalls;
        }
        else if (Array.isArray(parsedOutput)) {
            // Vertex and AIS Format
            functionCalls = parsedOutput
                .filter((obj) => Object.prototype.hasOwnProperty.call(obj, 'functionCall'))
                .map((obj) => obj.functionCall);
        }
        else {
            throw new Error();
        }
    }
    catch {
        throw new Error(`Google did not return a valid-looking function call: ${JSON.stringify(output)}`);
    }
    const interpolatedFunctions = loadFile(functions, vars);
    for (const functionCall of functionCalls) {
        // Parse function call and validate it against schema
        const functionName = functionCall.name;
        const functionArgs = parseStringObject(functionCall.args);
        const functionDeclarations = interpolatedFunctions?.find((f) => 'functionDeclarations' in f);
        const functionSchema = functionDeclarations?.functionDeclarations?.find((f) => f.name === functionName);
        if (!functionSchema) {
            throw new Error(`Called "${functionName}", but there is no function with that name`);
        }
        if (Object.keys(functionArgs).length !== 0 && functionSchema?.parameters) {
            const parameterSchema = normalizeSchemaTypes(functionSchema.parameters);
            let validate;
            try {
                validate = ajv.compile(parameterSchema);
            }
            catch (err) {
                throw new Error(`Tool schema doesn't compile with ajv: ${err}. If this is a valid tool schema you may need to reformulate your assertion without is-valid-function-call.`);
            }
            if (!validate(functionArgs)) {
                throw new Error(`Call to "${functionName}":\n${JSON.stringify(functionCall)}\ndoes not match schema:\n${JSON.stringify(validate.errors)}`);
            }
        }
        else if (!(JSON.stringify(functionArgs) === '{}' && !functionSchema?.parameters)) {
            throw new Error(`Call to "${functionName}":\n${JSON.stringify(functionCall)}\ndoes not match schema:\n${JSON.stringify(functionSchema)}`);
        }
    }
}
//# sourceMappingURL=util.js.map