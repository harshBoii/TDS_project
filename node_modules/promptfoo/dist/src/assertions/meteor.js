"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.handleMeteorAssertion = handleMeteorAssertion;
const natural_1 = require("natural");
const invariant_1 = __importDefault(require("../util/invariant"));
function preprocessWord(word) {
    return word.toLowerCase();
}
function generateEnums(candidate, reference, preprocess = preprocessWord) {
    if (typeof candidate === 'string') {
        throw new TypeError(`"candidate" expects pre-tokenized candidate (string[]): ${candidate}`);
    }
    if (typeof reference === 'string') {
        throw new TypeError(`"reference" expects pre-tokenized reference (string[]): ${reference}`);
    }
    const enumCandidateList = candidate.map((word, idx) => [idx, preprocess(word)]);
    const enumReferenceList = reference.map((word, idx) => [idx, preprocess(word)]);
    return [enumCandidateList, enumReferenceList];
}
function matchExactEnums(enumCandidateList, enumReferenceList) {
    const wordMatch = [];
    const candidateCopy = [...enumCandidateList];
    const referenceCopy = [...enumReferenceList];
    for (let i = candidateCopy.length - 1; i >= 0; i--) {
        for (let j = referenceCopy.length - 1; j >= 0; j--) {
            if (candidateCopy[i][1] === referenceCopy[j][1]) {
                wordMatch.push([candidateCopy[i][0], referenceCopy[j][0]]);
                candidateCopy.splice(i, 1);
                referenceCopy.splice(j, 1);
                break;
            }
        }
    }
    return [wordMatch, candidateCopy, referenceCopy];
}
function matchStemEnums(enumCandidateList, enumReferenceList, stemmer = natural_1.PorterStemmer) {
    const candidateCopy = [...enumCandidateList];
    const referenceCopy = [...enumReferenceList];
    // Create stemmed versions of words
    const candidateStems = candidateCopy.map(([idx, word]) => [idx, stemmer.stem(word)]);
    const referenceStems = referenceCopy.map(([idx, word]) => [idx, stemmer.stem(word)]);
    return matchExactEnums(candidateStems.map(([idx, stem]) => [idx, stem]), referenceStems.map(([idx, stem]) => [idx, stem]));
}
async function matchSynonymEnums(enumCandidateList, enumReferenceList, wordnet = new natural_1.WordNet()) {
    const wordMatch = [];
    const candidateCopy = [...enumCandidateList];
    const referenceCopy = [...enumReferenceList];
    for (let i = candidateCopy.length - 1; i >= 0; i--) {
        const candidateWord = candidateCopy[i][1];
        // Get all synsets and their synonyms
        const candidateSynsets = await new Promise((resolve) => {
            wordnet.lookup(candidateWord, (results) => resolve(results));
        });
        // Create set of synonyms, filtering out ones with underscores
        // and including the original word
        const candidateSynonymSet = new Set([
            candidateWord,
            ...candidateSynsets.flatMap((synset) => synset.synonyms.filter((syn) => !syn.includes('_'))),
        ]);
        for (let j = referenceCopy.length - 1; j >= 0; j--) {
            const referenceWord = referenceCopy[j][1];
            if (candidateSynonymSet.has(referenceWord)) {
                wordMatch.push([candidateCopy[i][0], referenceCopy[j][0]]);
                candidateCopy.splice(i, 1);
                referenceCopy.splice(j, 1);
                break;
            }
        }
    }
    return [wordMatch, candidateCopy, referenceCopy];
}
function countChunks(matches) {
    if (matches.length === 0) {
        return 0;
    }
    let chunks = 1;
    for (let i = 0; i < matches.length - 1; i++) {
        if (matches[i + 1][0] !== matches[i][0] + 1 || matches[i + 1][1] !== matches[i][1] + 1) {
            chunks++;
        }
    }
    return chunks;
}
async function calculateSingleMeteorScore(reference, candidate, alpha = 0.9, beta = 3.0, gamma = 0.5) {
    const [enumCandidate, enumReference] = generateEnums(candidate, reference);
    const translationLength = enumCandidate.length;
    const referenceLength = enumReference.length;
    // Stage 1: Exact matches
    const [exactMatches, remainingCandidate, remainingReference] = matchExactEnums(enumCandidate, enumReference);
    // Stage 2: Stem matches
    const [stemMatches, remainingCandidateAfterStem, remainingReferenceAfterStem] = matchStemEnums(remainingCandidate, remainingReference);
    // Stage 3: Synonym matches
    const [synonymMatches, ,] = await matchSynonymEnums(remainingCandidateAfterStem, remainingReferenceAfterStem);
    // Combine all matches
    const allMatches = [...exactMatches, ...stemMatches, ...synonymMatches].sort((a, b) => a[0] - b[0]);
    const matchesCount = allMatches.length;
    if (matchesCount === 0) {
        return 0;
    }
    let fragFrac = 0;
    let fmean = 0;
    if (translationLength === 0 || referenceLength === 0 || matchesCount === 0) {
        return 0.0;
    }
    const precision = matchesCount / translationLength;
    const recall = matchesCount / referenceLength;
    const denominator = alpha * precision + (1 - alpha) * recall;
    if (denominator === 0) {
        return 0.0;
    }
    fmean = (precision * recall) / denominator;
    const chunkCount = countChunks(allMatches);
    fragFrac = chunkCount / matchesCount;
    const penalty = gamma * Math.pow(fragFrac, beta);
    return (1 - penalty) * fmean;
}
async function calculateMeteorScore(candidate, references, alpha = 0.9, beta = 3.0, gamma = 0.5) {
    if (!candidate || references.length === 0) {
        throw new Error('Invalid inputs');
    }
    const scores = await Promise.all(references.map((reference) => calculateSingleMeteorScore(reference.split(/\s+/).map((word) => word.replace(/\.+$/, '')), candidate.split(/\s+/).map((word) => word.replace(/\.+$/, '')), alpha, beta, gamma)));
    return Math.max(...scores);
}
async function handleMeteorAssertion({ assertion, inverse, outputString, renderedValue, test, }) {
    // Validate inputs
    (0, invariant_1.default)(typeof renderedValue === 'string' ||
        (Array.isArray(renderedValue) && renderedValue.every((v) => typeof v === 'string')), '"meteor" assertion must have a string or array of strings value');
    const references = Array.isArray(renderedValue) ? renderedValue : [renderedValue];
    // Get parameters from assertion or use defaults
    const meteorAssertion = assertion;
    const alpha = meteorAssertion.alpha ?? 0.9;
    const beta = meteorAssertion.beta ?? 3.0;
    const gamma = meteorAssertion.gamma ?? 0.5;
    const threshold = meteorAssertion.threshold ?? 0.5;
    const score = await calculateMeteorScore(outputString, references, alpha, beta, gamma);
    const pass = inverse ? score < threshold : score >= threshold;
    return {
        pass,
        score: inverse ? 1 - score : score,
        reason: pass
            ? 'METEOR assertion passed'
            : `METEOR score ${score.toFixed(4)} did not meet threshold ${threshold}`,
        assertion,
    };
}
//# sourceMappingURL=meteor.js.map