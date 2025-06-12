"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkRemoteHealth = checkRemoteHealth;
const envars_1 = require("../envars");
const fetch_1 = require("../fetch");
const cloud_1 = require("../globalConfig/cloud");
const logger_1 = __importDefault(require("../logger"));
/**
 * Checks the health of the remote API.
 * @param url - The URL to check.
 * @returns A promise that resolves to the health check response.
 */
async function checkRemoteHealth(url) {
    logger_1.default.debug(`[CheckRemoteHealth] Checking API health: ${JSON.stringify({
        url,
        // Log environment variables that might affect network requests
        env: {
            httpProxy: (0, envars_1.getEnvString)('HTTP_PROXY') || (0, envars_1.getEnvString)('http_proxy'),
            httpsProxy: (0, envars_1.getEnvString)('HTTPS_PROXY') || (0, envars_1.getEnvString)('https_proxy'),
            allProxy: (0, envars_1.getEnvString)('ALL_PROXY') || (0, envars_1.getEnvString)('all_proxy'),
            noProxy: (0, envars_1.getEnvString)('NO_PROXY') || (0, envars_1.getEnvString)('no_proxy'),
            nodeExtra: (0, envars_1.getEnvString)('NODE_EXTRA_CA_CERTS'),
            nodeTls: (0, envars_1.getEnvString)('NODE_TLS_REJECT_UNAUTHORIZED'),
        },
    })}`);
    try {
        const cloudConfig = new cloud_1.CloudConfig();
        const requestOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };
        logger_1.default.debug(`[CheckRemoteHealth] Making fetch request: ${JSON.stringify({
            url,
            options: requestOptions,
            timeout: 5000,
            nodeVersion: process.version,
        })}`);
        const response = await (0, fetch_1.fetchWithTimeout)(url, requestOptions, 5000);
        if (!response.ok) {
            logger_1.default.debug(`[CheckRemoteHealth] API health check failed with non-OK response: ${JSON.stringify({
                status: response.status,
                statusText: response.statusText,
                url,
            })}`);
            return {
                status: 'ERROR',
                message: `Failed to connect: ${response.status} ${response.statusText}`,
            };
        }
        const data = await response.json();
        logger_1.default.debug(`[CheckRemoteHealth] API health check response: ${JSON.stringify({ data })}`);
        if (data.status === 'OK') {
            return {
                status: 'OK',
                message: cloudConfig.isEnabled()
                    ? 'Cloud API is healthy (using custom endpoint)'
                    : 'Cloud API is healthy',
            };
        }
        if (data.status === 'DISABLED') {
            return {
                status: 'DISABLED',
                message: 'remote generation and grading are disabled',
            };
        }
        return {
            status: 'ERROR',
            message: data.message || 'Unknown error',
        };
    }
    catch (err) {
        // Type guard for Error objects
        const error = err instanceof Error ? err : new Error(String(err));
        // If it's a timeout error, return a softer message
        if (error.name === 'TimeoutError') {
            return {
                status: 'OK',
                message: 'API health check timed out, proceeding anyway',
            };
        }
        // Handle certificate errors specifically
        if (error.message.includes('certificate')) {
            return {
                status: 'ERROR',
                message: `Network error: SSL/Certificate issue detected - ${error.message}`,
            };
        }
        // For other network errors, include more details including the cause if available
        const cause = 'cause' in error ? ` (Cause: ${error.cause})` : '';
        const code = 'code' in error ? ` [${error['code']}]` : '';
        logger_1.default.debug(`[CheckRemoteHealth] API health check failed: ${JSON.stringify({
            error: error.message,
            url,
        })}`);
        return {
            status: 'ERROR',
            message: `Network error${code}: ${error.message}${cause}\nURL: ${url}`,
        };
    }
}
//# sourceMappingURL=apiHealth.js.map