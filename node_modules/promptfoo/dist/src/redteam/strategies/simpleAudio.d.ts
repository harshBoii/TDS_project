import type { TestCase } from '../../types';
/**
 * Converts text to audio using the remote API
 * @throws Error if remote generation is disabled or if the API call fails
 */
export declare function textToAudio(text: string, language?: string): Promise<string>;
/**
 * Adds audio encoding to test cases
 * @throws Error if the remote API for audio conversion is unavailable
 */
export declare function addAudioToBase64(testCases: TestCase[], injectVar: string, config?: Record<string, any>): Promise<TestCase[]>;
//# sourceMappingURL=simpleAudio.d.ts.map