"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.handlePiScorer = void 0;
const matchers_1 = require("../matchers");
const invariant_1 = __importDefault(require("../util/invariant"));
const handlePiScorer = async ({ assertion, prompt, renderedValue, outputString, }) => {
    (0, invariant_1.default)(typeof renderedValue === 'string', '"pi" assertion type must have a string value');
    (0, invariant_1.default)(typeof prompt === 'string', '"pi" assertion must have a prompt that is a string');
    return (0, matchers_1.matchesPiScore)(renderedValue, prompt, outputString, assertion);
};
exports.handlePiScorer = handlePiScorer;
//# sourceMappingURL=pi.js.map