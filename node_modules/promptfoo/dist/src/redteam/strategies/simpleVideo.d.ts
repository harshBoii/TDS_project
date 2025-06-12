import type { TestCase } from '../../types';
export declare function importFfmpeg(): Promise<any>;
export declare function createTempVideoEnvironment(text: string): Promise<{
    tempDir: string;
    textFilePath: string;
    outputPath: string;
    cleanup: () => void;
}>;
export declare function getFallbackBase64(text: string): string;
export declare function textToVideo(text: string): Promise<string>;
export declare function createProgressBar(total: number): {
    increment: () => void;
    stop: () => void;
};
export declare function addVideoToBase64(testCases: TestCase[], injectVar: string, videoGenerator?: (text: string) => Promise<string>): Promise<TestCase[]>;
export declare function writeVideoFile(base64Video: string, outputFilePath: string): Promise<void>;
export declare function main(): Promise<void>;
//# sourceMappingURL=simpleVideo.d.ts.map