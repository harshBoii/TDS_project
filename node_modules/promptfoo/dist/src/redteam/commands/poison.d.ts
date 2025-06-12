import type { Command } from 'commander';
interface PoisonOptions {
    documents: string[];
    goal?: string;
    output?: string;
    outputDir?: string;
    envPath?: string;
    verbose?: boolean;
}
interface PoisonResponse {
    poisonedDocument: string;
    intendedResult: string;
    task: string;
    originalPath?: string;
}
type FilePath = string;
type DocumentContent = string;
type Document = {
    docLike: FilePath | DocumentContent;
    isFile: boolean;
    dir: string | null;
};
export declare function getAllFiles(dirPath: string, arrayOfFiles?: string[]): string[];
export declare function generatePoisonedDocument(document: string, goal?: PoisonOptions['goal']): Promise<PoisonResponse>;
/**
 * Poisons an individual document.
 * @param docLike A path to a document or document content to poison.
 */
export declare function poisonDocument(doc: Document, outputDir: string, goal?: PoisonOptions['goal']): Promise<Partial<PoisonResponse>>;
export declare function doPoisonDocuments(options: PoisonOptions): Promise<void>;
export declare function poisonCommand(program: Command): void;
export {};
//# sourceMappingURL=poison.d.ts.map