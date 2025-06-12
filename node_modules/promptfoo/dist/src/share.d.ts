import type Eval from './models/eval';
export interface ShareDomainResult {
    domain: string;
    isPublicShare: boolean;
}
export declare function isSharingEnabled(evalRecord: Eval): boolean;
export declare function determineShareDomain(eval_: Eval): ShareDomainResult;
/**
 * Removes authentication information (username and password) from a URL.
 *
 * This function addresses a security concern raised in GitHub issue #1184,
 * where sensitive authentication information was being displayed in the CLI output.
 * By default, we now strip this information to prevent accidental exposure of credentials.
 *
 * @param urlString - The URL string that may contain authentication information.
 * @returns A new URL string with username and password removed, if present.
 *          If URL parsing fails, it returns the original string.
 */
export declare function stripAuthFromUrl(urlString: string): string;
/**
 * Constructs the shareable URL for an eval.
 * @param eval_ The eval to get the shareable URL for.
 * @param showAuth Whether to show the authentication information in the URL.
 * @returns The shareable URL for the eval.
 */
export declare function getShareableUrl(eval_: Eval, showAuth?: boolean): Promise<string | null>;
/**
 * Shares an eval and returns the shareable URL.
 * @param evalRecord The eval to share.
 * @param showAuth Whether to show the authentication information in the URL.
 * @returns The shareable URL for the eval.
 */
export declare function createShareableUrl(evalRecord: Eval, showAuth?: boolean): Promise<string | null>;
/**
 * Checks whether an eval has been shared.
 * @param eval_ The eval to check.
 * @returns True if the eval has been shared, false otherwise.
 */
export declare function hasEvalBeenShared(eval_: Eval): Promise<boolean>;
//# sourceMappingURL=share.d.ts.map