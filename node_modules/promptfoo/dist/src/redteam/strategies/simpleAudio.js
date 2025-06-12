"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.textToAudio = textToAudio;
exports.addAudioToBase64 = addAudioToBase64;
const cli_progress_1 = require("cli-progress");
const cache_1 = require("../../cache");
const constants_1 = require("../../constants");
const accounts_1 = require("../../globalConfig/accounts");
const logger_1 = __importDefault(require("../../logger"));
const shared_1 = require("../../providers/shared");
const invariant_1 = __importDefault(require("../../util/invariant"));
const remoteGeneration_1 = require("../remoteGeneration");
/**
 * Converts text to audio using the remote API
 * @throws Error if remote generation is disabled or if the API call fails
 */
async function textToAudio(text, language = 'en') {
    // Check if remote generation is disabled
    if ((0, remoteGeneration_1.neverGenerateRemote)()) {
        throw new Error('Remote generation is disabled but required for audio strategy. Please enable remote generation to use this strategy.');
    }
    try {
        logger_1.default.debug(`Using remote generation for audio task`);
        const payload = {
            task: 'audio',
            text,
            language,
            version: constants_1.VERSION,
            email: (0, accounts_1.getUserEmail)(),
        };
        const { data } = await (0, cache_1.fetchWithCache)((0, remoteGeneration_1.getRemoteGenerationUrl)(), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        }, shared_1.REQUEST_TIMEOUT_MS);
        if (data.error) {
            throw new Error(`Error in remote audio generation: ${data.error}`);
        }
        logger_1.default.debug(`Received audio base64 from remote API (${data.audioBase64.length} chars)`);
        return data.audioBase64;
    }
    catch (error) {
        logger_1.default.error(`Error generating audio from text: ${error}`);
        throw new Error(`Failed to generate audio: ${error instanceof Error ? error.message : String(error)}. This strategy requires an active internet connection and access to the remote API.`);
    }
}
/**
 * Adds audio encoding to test cases
 * @throws Error if the remote API for audio conversion is unavailable
 */
async function addAudioToBase64(testCases, injectVar, config = {}) {
    const audioTestCases = [];
    const language = config.language || 'en';
    let progressBar;
    if (logger_1.default.level !== 'debug') {
        progressBar = new cli_progress_1.SingleBar({
            format: 'Converting to Audio {bar} {percentage}% | ETA: {eta}s | {value}/{total}',
            hideCursor: true,
            gracefulExit: true,
        }, cli_progress_1.Presets.shades_classic);
        progressBar.start(testCases.length, 0);
    }
    for (const testCase of testCases) {
        (0, invariant_1.default)(testCase.vars, `Audio encoding: testCase.vars is required, but got ${JSON.stringify(testCase)}`);
        const originalText = String(testCase.vars[injectVar]);
        // Convert text to audio using the remote API
        const base64Audio = await textToAudio(originalText, language);
        audioTestCases.push({
            ...testCase,
            assert: testCase.assert?.map((assertion) => ({
                ...assertion,
                metric: assertion.type?.startsWith('promptfoo:redteam:')
                    ? `${assertion.type?.split(':').pop() || assertion.metric}/Audio-Encoded`
                    : assertion.metric,
            })),
            vars: {
                ...testCase.vars,
                [injectVar]: base64Audio,
            },
            metadata: {
                ...testCase.metadata,
                strategyId: 'audio',
                originalText,
            },
        });
        if (progressBar) {
            progressBar.increment(1);
        }
        else {
            logger_1.default.debug(`Processed ${audioTestCases.length} of ${testCases.length}`);
        }
    }
    if (progressBar) {
        progressBar.stop();
    }
    return audioTestCases;
}
//# sourceMappingURL=simpleAudio.js.map