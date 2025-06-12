"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.processCsvPrompts = processCsvPrompts;
const sync_1 = require("csv-parse/sync");
const fs_1 = __importDefault(require("fs"));
const envars_1 = require("../../envars");
/**
 * Process a CSV file containing prompts
 *
 * CSV format can be either:
 * 1. Single column with prompt text per line
 * 2. CSV with a 'prompt' column and optional 'label' column
 *
 * @param filePath Path to the CSV file
 * @param basePrompt Base prompt properties to include
 * @returns Array of processed prompts
 */
async function processCsvPrompts(filePath, basePrompt) {
    const content = fs_1.default.readFileSync(filePath, 'utf8');
    const delimiter = (0, envars_1.getEnvString)('PROMPTFOO_CSV_DELIMITER', ',');
    const enforceStrict = (0, envars_1.getEnvBool)('PROMPTFOO_CSV_STRICT', false);
    if (!content.includes(delimiter)) {
        const lines = content.split(/\r?\n/).filter((line) => line.trim());
        const startIndex = lines[0]?.toLowerCase().trim() === 'prompt' ? 1 : 0;
        return lines.slice(startIndex).map((line, index) => ({
            ...basePrompt,
            raw: line,
            label: basePrompt.label || `Prompt ${index + 1} - ${line}`,
        }));
    }
    try {
        const parseOptions = {
            columns: true,
            bom: true,
            delimiter,
            relax_quotes: !enforceStrict,
            skip_empty_lines: true,
            trim: true,
        };
        const records = (0, sync_1.parse)(content, parseOptions);
        return records
            .filter((row) => row.prompt)
            .map((row, index) => {
            return {
                ...basePrompt,
                raw: row.prompt,
                label: row.label || basePrompt.label || `Prompt ${index + 1} - ${row.prompt}`,
            };
        });
    }
    catch {
        const lines = content.split(/\r?\n/).filter((line) => line.trim());
        const startIndex = lines[0]?.toLowerCase().trim() === 'prompt' ? 1 : 0;
        return lines.slice(startIndex).map((line, index) => ({
            ...basePrompt,
            raw: line,
            label: basePrompt.label || `Prompt ${index + 1} - ${line}`,
        }));
    }
}
//# sourceMappingURL=csv.js.map