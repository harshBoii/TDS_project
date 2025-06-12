import readline from 'readline';
/**
 * Factory function for creating readline interface.
 * This abstraction makes it easier to mock in tests and prevents open handles.
 */
export declare function createReadlineInterface(): readline.Interface;
/**
 * Prompts the user with a question and returns their answer.
 * Automatically handles cleanup of the readline interface.
 */
export declare function promptUser(question: string): Promise<string>;
/**
 * Prompts the user with a yes/no question and returns a boolean.
 * @param question The question to ask
 * @param defaultYes If true, empty response defaults to yes. If false, defaults to no.
 */
export declare function promptYesNo(question: string, defaultYes?: boolean): Promise<boolean>;
//# sourceMappingURL=readline.d.ts.map