/**
 * Loads the content from a file reference
 * @param fileRef The file reference string (e.g. 'file://path/to/file.json')
 * @param basePath Base path for resolving relative paths
 * @returns The loaded content from the file
 */
export declare function loadFileReference(fileRef: string, basePath?: string): Promise<any>;
/**
 * Recursively processes a configuration object, replacing any file:// references
 * with the content of the referenced files
 * @param config The configuration object to process
 * @param basePath Base path for resolving relative paths
 * @returns A new configuration object with file references resolved
 */
export declare function processConfigFileReferences(config: any, basePath?: string): Promise<any>;
//# sourceMappingURL=fileReference.d.ts.map