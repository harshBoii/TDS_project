import type { Prompt } from '../../types';
/**
 * Process a CSV file containing prompts
 *
 * CSV format can be either:
 * 1. Single column with prompt text per line
 * 2. CSV with a 'prompt' column and optional 'label' column
 *
 * @param filePath Path to the CSV file
 * @param basePrompt Base prompt properties to include
 * @returns Array of processed prompts
 */
export declare function processCsvPrompts(filePath: string, basePrompt: Partial<Prompt>): Promise<Prompt[]>;
//# sourceMappingURL=csv.d.ts.map