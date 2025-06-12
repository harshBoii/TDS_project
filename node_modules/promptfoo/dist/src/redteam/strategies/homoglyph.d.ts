import type { TestCase } from '../../types';
export declare const homoglyphMap: {
    [key: string]: string;
};
/**
 * Convert text to homoglyphs (visually similar Unicode characters)
 */
export declare function toHomoglyphs(text: string): string;
/**
 * Add homoglyph encoding to test cases
 */
export declare function addHomoglyphs(testCases: TestCase[], injectVar: string): TestCase[];
//# sourceMappingURL=homoglyph.d.ts.map