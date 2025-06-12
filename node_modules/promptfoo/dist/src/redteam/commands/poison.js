"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAllFiles = getAllFiles;
exports.generatePoisonedDocument = generatePoisonedDocument;
exports.poisonDocument = poisonDocument;
exports.doPoisonDocuments = doPoisonDocuments;
exports.poisonCommand = poisonCommand;
const chalk_1 = __importDefault(require("chalk"));
const fs = __importStar(require("fs"));
const js_yaml_1 = __importDefault(require("js-yaml"));
const path = __importStar(require("path"));
const accounts_1 = require("../../globalConfig/accounts");
const logger_1 = __importDefault(require("../../logger"));
const telemetry_1 = __importDefault(require("../../telemetry"));
const util_1 = require("../../util");
const remoteGeneration_1 = require("../remoteGeneration");
function getAllFiles(dirPath, arrayOfFiles = []) {
    const files = fs.readdirSync(dirPath);
    files.forEach((file) => {
        const fullPath = path.join(dirPath, file);
        if (fs.statSync(fullPath).isDirectory()) {
            arrayOfFiles = getAllFiles(fullPath, arrayOfFiles);
        }
        else {
            arrayOfFiles.push(fullPath);
        }
    });
    return arrayOfFiles;
}
async function generatePoisonedDocument(document, goal) {
    const response = await fetch((0, remoteGeneration_1.getRemoteGenerationUrl)(), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            task: 'poison-document',
            document,
            goal,
            email: (0, accounts_1.getUserEmail)(),
        }),
    });
    if (!response.ok) {
        throw new Error(`Failed to generate poisoned document, ${response.status} ${response.statusText}: ${await response.text()}`);
    }
    return await response.json();
}
/**
 * Poisons an individual document.
 * @param docLike A path to a document or document content to poison.
 */
async function poisonDocument(doc, outputDir, goal) {
    logger_1.default.debug(`Poisoning ${JSON.stringify(doc)}`);
    try {
        const documentContent = doc.isFile ? fs.readFileSync(doc.docLike, 'utf-8') : doc.docLike;
        const result = await generatePoisonedDocument(documentContent, goal);
        if (doc.isFile) {
            result.originalPath = doc.docLike;
        }
        let outputFilePath;
        if (doc.isFile) {
            // If the document was loaded from a directory, strip the directory prefix.
            // Otherwise, use the relative path to the current directory.
            const docPath = doc.dir
                ? doc.docLike.replace(doc.dir, '')
                : path.relative(process.cwd(), doc.docLike);
            outputFilePath = path.join(outputDir, docPath);
            // Create necessary subdirectories
            fs.mkdirSync(path.dirname(outputFilePath), { recursive: true });
        }
        else {
            // Generate a filename for / from the document content
            const hash = Buffer.from(documentContent).toString('base64').slice(0, 8);
            outputFilePath = path.join(outputDir, `poisoned-${hash}.txt`);
        }
        fs.writeFileSync(outputFilePath, result.poisonedDocument);
        logger_1.default.debug(`Wrote poisoned document to ${outputFilePath}`);
        logger_1.default.info(chalk_1.default.green(`✓ Successfully poisoned ${doc.isFile ? doc.docLike : 'document'}`));
        return {
            originalPath: result.originalPath,
            poisonedDocument: result.poisonedDocument,
            intendedResult: result.intendedResult,
        };
    }
    catch (error) {
        throw new Error(`Failed to poison ${doc.docLike}: ${error}`);
    }
}
async function doPoisonDocuments(options) {
    const outputPath = options.output || 'poisoned-config.yaml';
    const outputDir = options.outputDir || 'poisoned-documents';
    // Always create output directory since we're always using it
    fs.mkdirSync(outputDir, { recursive: true });
    logger_1.default.info(chalk_1.default.blue('Generating poisoned documents...'));
    // Collect all document paths, including from directories
    let docs = [];
    for (const doc of options.documents) {
        // Is the document a ∈{file|directory} path or document content?
        if (fs.existsSync(doc)) {
            const stat = fs.statSync(doc);
            if (stat.isDirectory()) {
                docs = [
                    ...docs,
                    ...getAllFiles(doc).map((file) => ({ docLike: file, isFile: true, dir: doc })),
                ];
            }
            else {
                docs = [...docs, { docLike: doc, isFile: true, dir: null }];
            }
        }
        else {
            // Treat as direct content
            docs.push({ docLike: doc, isFile: false, dir: null });
        }
    }
    // Poison all documents
    const results = await Promise.all(docs.map((doc) => poisonDocument(doc, outputDir, options.goal)));
    // Write summary YAML file
    fs.writeFileSync(outputPath, js_yaml_1.default.dump({ documents: results }));
    logger_1.default.info(chalk_1.default.green(`\nWrote ${results.length} poisoned documents summary to ${outputPath}`));
}
function poisonCommand(program) {
    program
        .command('poison')
        .description('Generate poisoned documents for RAG testing')
        .argument('<documents...>', 'Documents, directories, or text content to poison')
        .option('-g, --goal <goal>', 'Goal/intended result of the poisoning')
        .option('-o, --output <path>', 'Output YAML file path', 'poisoned-config.yaml')
        .option('-d, --output-dir <path>', 'Directory to write individual poisoned documents', 'poisoned-documents')
        .option('--env-file, --env-path <path>', 'Path to .env file')
        .action(async (documents, opts) => {
        (0, util_1.setupEnv)(opts.envPath);
        telemetry_1.default.record('command_used', {
            name: 'redteam poison',
        });
        await telemetry_1.default.send();
        try {
            await doPoisonDocuments({
                documents,
                ...opts,
            });
        }
        catch (error) {
            logger_1.default.error(`An unexpected error occurred: ${error}`);
            process.exit(1);
        }
    });
}
//# sourceMappingURL=poison.js.map