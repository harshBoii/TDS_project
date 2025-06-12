import type { AssertionParams, GradingResult } from '../types';
/**
 * Calculates the Google-BLEU (GLEU) score for a candidate string against reference strings.
 *
 * GLEU is a variant of BLEU that shows better correlation with human judgments on sentence-level
 * evaluation. It calculates n-gram matches between the candidate and reference texts.
 *
 * For the GLEU score, we record all sub-sequences of 1, 2, 3 or 4 tokens in output and target sequence.
 * We then compute:
 * - Precision: the ratio of matching n-grams to total n-grams in the generated output sequence
 * - Recall: the ratio of matching n-grams to total n-grams in the target (ground truth) sequence
 *
 * The GLEU score is the minimum of precision and recall.
 *
 * For multiple references, we calculate the GLEU score against each reference and return the maximum score.
 * This reflects the intuition that if the candidate matches well with any of the valid references,
 * it should be considered a good translation.
 *
 * Implementation details:
 * - n-grams from n=1 to n=4 are considered by default
 * - The calculation is case-insensitive
 * - Identical strings will always receive a score of 1
 * - If there are no n-grams in common, the score will be 0
 *
 * @param candidate - The string to evaluate
 * @param references - Array of reference strings to compare against
 * @param minN - Minimum n-gram length to consider (default: 1)
 * @param maxN - Maximum n-gram length to consider (default: 4)
 * @returns GLEU score between 0 and 1, where higher scores indicate better matches
 * @throws When candidate or references are invalid
 */
export declare function calculateGleuScore(candidate: string, references: string[], minN?: number, maxN?: number): number;
/**
 * Handles GLEU (Google-BLEU) score calculation and evaluation for assertions.
 * GLEU is a variant of BLEU that correlates better with human judgments on sentence-level evaluation.
 *
 * Use cases for GLEU:
 * - For sentence-level evaluation where BLEU might give unintuitive results
 * - When you want to balance both precision and recall in your evaluation
 * - When working with multiple valid references
 * - When human correlation is particularly important
 *
 * @param {AssertionParams} params - The parameters for GLEU score evaluation
 * @param {Object} params.assertion - The assertion configuration object
 * @param {boolean} params.inverse - Whether to invert the pass condition
 * @param {string} params.outputString - The candidate string to evaluate
 * @param {string|string[]} params.renderedValue - The reference string(s) to compare against
 * @param {string} params.provider - The provider name (unused)
 * @param {Object} params.test - The test case data (unused)
 * @returns {GradingResult} Object containing:
 *   - pass: boolean indicating if assertion passed
 *   - score: GLEU score (0-1)
 *   - reason: explanation of the result
 *   - assertion: original assertion config
 * @throws {Error} If renderedValue is not a string or array of strings
 */
export declare function handleGleuScore({ assertion, inverse, outputString, renderedValue, provider, test, }: AssertionParams): GradingResult;
//# sourceMappingURL=gleu.d.ts.map