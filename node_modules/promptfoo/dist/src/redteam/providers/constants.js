"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TEMPERATURE = exports.ATTACKER_MODEL_SMALL = exports.ATTACKER_MODEL = void 0;
const envars_1 = require("../../envars");
exports.ATTACKER_MODEL = 'gpt-4.1-2025-04-14';
exports.ATTACKER_MODEL_SMALL = 'gpt-4o-mini';
exports.TEMPERATURE = (0, envars_1.getEnvFloat)('PROMPTFOO_JAILBREAK_TEMPERATURE')
    ? (0, envars_1.getEnvFloat)('PROMPTFOO_JAILBREAK_TEMPERATURE')
    : 0.7;
//# sourceMappingURL=constants.js.map