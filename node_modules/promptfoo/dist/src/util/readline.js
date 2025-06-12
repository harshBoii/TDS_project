"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.createReadlineInterface = createReadlineInterface;
exports.promptUser = promptUser;
exports.promptYesNo = promptYesNo;
const readline_1 = __importDefault(require("readline"));
/**
 * Factory function for creating readline interface.
 * This abstraction makes it easier to mock in tests and prevents open handles.
 */
function createReadlineInterface() {
    return readline_1.default.createInterface({
        input: process.stdin,
        output: process.stdout,
    });
}
/**
 * Prompts the user with a question and returns their answer.
 * Automatically handles cleanup of the readline interface.
 */
async function promptUser(question) {
    return new Promise((resolve, reject) => {
        let rl = null;
        try {
            rl = createReadlineInterface();
            // Handle errors
            rl.on('error', (err) => {
                if (rl) {
                    rl.close();
                }
                reject(err);
            });
            rl.question(question, (answer) => {
                if (rl) {
                    rl.close();
                }
                resolve(answer);
            });
        }
        catch (err) {
            if (rl) {
                rl.close();
            }
            reject(err);
        }
    });
}
/**
 * Prompts the user with a yes/no question and returns a boolean.
 * @param question The question to ask
 * @param defaultYes If true, empty response defaults to yes. If false, defaults to no.
 */
async function promptYesNo(question, defaultYes = false) {
    const suffix = defaultYes ? '(Y/n): ' : '(y/N): ';
    const answer = await promptUser(`${question} ${suffix}`);
    if (defaultYes) {
        return !answer.trim().toLowerCase().startsWith('n');
    }
    return answer.trim().toLowerCase().startsWith('y');
}
//# sourceMappingURL=readline.js.map