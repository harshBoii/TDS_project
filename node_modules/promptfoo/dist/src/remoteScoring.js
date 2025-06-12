"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getWithPiApiKey = getWithPiApiKey;
exports.doRemoteScoringWithPi = doRemoteScoringWithPi;
const cache_1 = require("./cache");
const envars_1 = require("./envars");
const logger_1 = __importDefault(require("./logger"));
const shared_1 = require("./providers/shared");
function getWithPiApiKey() {
    // Check env var first
    const withPiApiKey = (0, envars_1.getEnvString)('WITHPI_API_KEY');
    if (withPiApiKey) {
        return withPiApiKey;
    }
}
function convertPiResultToGradingResult(result, threshold) {
    return {
        pass: result.total_score > threshold,
        score: result.total_score,
        namedScores: result.question_scores,
        reason: 'Pi Scorer',
    };
}
const WITHPI_API_URL = `https://api.withpi.ai/v1/scoring_system/score`;
async function doRemoteScoringWithPi(payload, passThreshold = 0.5) {
    try {
        const apiKey = getWithPiApiKey();
        if (apiKey) {
            const body = JSON.stringify(payload);
            logger_1.default.debug(`Performing remote scoring with pi: ${body}`);
            const { data } = await (0, cache_1.fetchWithCache)(WITHPI_API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-key': apiKey,
                },
                body,
            }, shared_1.REQUEST_TIMEOUT_MS);
            return convertPiResultToGradingResult(data, passThreshold);
        }
        else {
            throw new Error(`Env var WITHPI_API_KEY must be set. Visit https://docs.withpi.ai for more information.`);
        }
    }
    catch (error) {
        throw new Error(`Could not perform remote grading: ${error}`);
    }
}
//# sourceMappingURL=remoteScoring.js.map