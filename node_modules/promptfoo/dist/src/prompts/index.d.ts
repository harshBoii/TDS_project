import type { UnifiedConfig, Prompt, TestSuite, EvaluateTestSuite } from '../types';
export * from './grading';
/**
 * Reads and maps provider prompts based on the configuration and parsed prompts.
 * @param config - The configuration object.
 * @param parsedPrompts - Array of parsed prompts.
 * @returns A map of provider IDs to their respective prompts.
 */
export declare function readProviderPromptMap(config: Pick<Partial<UnifiedConfig>, 'providers'>, parsedPrompts: Prompt[]): TestSuite['providerPromptMap'];
/**
 * Processes a raw prompt based on its content type and path.
 * @param prompt - The raw prompt data.
 * @param basePath - Base path for file resolution.
 * @param maxRecursionDepth - Maximum recursion depth for globbing.
 * @returns Promise resolving to an array of processed prompts.
 */
export declare function processPrompt(prompt: Partial<Prompt>, basePath?: string, maxRecursionDepth?: number): Promise<Prompt[]>;
/**
 * Reads and processes prompts from a specified path or glob pattern.
 * @param promptPathOrGlobs - The path or glob pattern.
 * @param basePath - Base path for file resolution.
 * @returns Promise resolving to an array of processed prompts.
 */
export declare function readPrompts(promptPathOrGlobs: string | (string | Partial<Prompt>)[] | Record<string, string>, basePath?: string): Promise<Prompt[]>;
export declare function processPrompts(prompts: EvaluateTestSuite['prompts']): Promise<TestSuite['prompts']>;
export declare const GEVAL_PROMPT_STEPS = "\nGiven an evaluation criteria which outlines how you should judge some text, generate 3-4 concise evaluation steps for any text based on the criteria below.\n\nEvaluation Criteria:\n{{criteria}}\n\n**\nIMPORTANT: Please make sure to only return in minified JSON format, with the \"steps\" key as a list of strings. No additional words, explanation or formatting is needed.\nExample JSON:\n{\"steps\": <list_of_strings>}\n**\n\nJSON:\n";
export declare const GEVAL_PROMPT_EVALUATE = "\nYou will be given one Reply for a Source Text below. Your task is to rate the Reply on one metric.\nPlease make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.\n\nEvaluation Criteria:\n{{criteria}}\n\nEvaluation Steps:\n- {{steps}}\n- Given the evaluation steps, return a JSON with two keys: 1) a \"score\" key ranging from 0 - {{maxScore}}, with {{maxScore}} being that it follows the Evaluation Criteria outlined in the Evaluation Steps and 0 being that it does not; 2) a \"reason\" key, a reason for the given score, but DO NOT QUOTE THE SCORE in your reason. Please mention specific information from Source Text and Reply in your reason, but be very concise with it!\n\nSource Text:\n{{input}}\n\nReply:\n{{output}}\n\n**\nIMPORTANT: Please make sure to only return in minified JSON format, with the \"score\" and \"reason\" key. No additional words, explanation or formatting is needed.\n\nExample JSON:\n{\"score\":0,\"reason\":\"The text does not follow the evaluation steps provided.\"}\n**\n\nJSON:\n";
//# sourceMappingURL=index.d.ts.map