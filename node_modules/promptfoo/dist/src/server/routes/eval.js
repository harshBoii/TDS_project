"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.evalJobs = exports.evalRouter = void 0;
const dedent_1 = __importDefault(require("dedent"));
const express_1 = require("express");
const uuid_1 = require("uuid");
const zod_1 = require("zod");
const zod_validation_error_1 = require("zod-validation-error");
const accounts_1 = require("../../globalConfig/accounts");
const index_1 = __importDefault(require("../../index"));
const logger_1 = __importDefault(require("../../logger"));
const eval_1 = __importDefault(require("../../models/eval"));
const evalResult_1 = __importDefault(require("../../models/evalResult"));
const database_1 = require("../../util/database");
const invariant_1 = __importDefault(require("../../util/invariant"));
const apiSchemas_1 = require("../apiSchemas");
exports.evalRouter = (0, express_1.Router)();
// Running jobs
exports.evalJobs = new Map();
exports.evalRouter.post('/job', (req, res) => {
    const { evaluateOptions, ...testSuite } = req.body;
    const id = (0, uuid_1.v4)();
    exports.evalJobs.set(id, {
        evalId: null,
        status: 'in-progress',
        progress: 0,
        total: 0,
        result: null,
        logs: [],
    });
    index_1.default
        .evaluate(Object.assign({}, testSuite, {
        writeLatestResults: true,
        sharing: testSuite.sharing ?? true,
    }), Object.assign({}, evaluateOptions, {
        eventSource: 'web',
        progressCallback: (progress, total) => {
            const job = exports.evalJobs.get(id);
            (0, invariant_1.default)(job, 'Job not found');
            job.progress = progress;
            job.total = total;
            console.log(`[${id}] ${progress}/${total}`);
        },
    }))
        .then(async (result) => {
        const job = exports.evalJobs.get(id);
        (0, invariant_1.default)(job, 'Job not found');
        job.status = 'complete';
        job.result = await result.toEvaluateSummary();
        job.evalId = result.id;
        console.log(`[${id}] Complete`);
    })
        .catch((error) => {
        logger_1.default.error((0, dedent_1.default) `Failed to eval tests:
        Error: ${error}
        Body: ${JSON.stringify(req.body, null, 2)}`);
        const job = exports.evalJobs.get(id);
        (0, invariant_1.default)(job, 'Job not found');
        job.status = 'error';
        job.result = null;
        job.evalId = null;
        job.logs = [String(error)];
    });
    res.json({ id });
});
exports.evalRouter.get('/job/:id', (req, res) => {
    const id = req.params.id;
    const job = exports.evalJobs.get(id);
    if (!job) {
        res.status(404).json({ error: 'Job not found' });
        return;
    }
    if (job.status === 'complete') {
        res.json({
            status: 'complete',
            result: job.result,
            evalId: job.evalId,
            logs: job.logs,
        });
    }
    else if (job.status === 'error') {
        res.json({
            status: 'error',
            logs: job.logs,
        });
    }
    else {
        res.json({
            status: 'in-progress',
            progress: job.progress,
            total: job.total,
            logs: job.logs,
        });
    }
});
exports.evalRouter.patch('/:id', (req, res) => {
    const id = req.params.id;
    const { table, config } = req.body;
    if (!id) {
        res.status(400).json({ error: 'Missing id' });
        return;
    }
    try {
        (0, database_1.updateResult)(id, config, table);
        res.json({ message: 'Eval updated successfully' });
    }
    catch {
        res.status(500).json({ error: 'Failed to update eval table' });
    }
});
exports.evalRouter.patch('/:id/author', async (req, res) => {
    try {
        const { id } = apiSchemas_1.ApiSchemas.Eval.UpdateAuthor.Params.parse(req.params);
        const { author } = apiSchemas_1.ApiSchemas.Eval.UpdateAuthor.Request.parse(req.body);
        const eval_ = await eval_1.default.findById(id);
        if (!eval_) {
            res.status(404).json({ error: 'Eval not found' });
            return;
        }
        if (!author) {
            res.status(400).json({ error: 'No author provided' });
            return;
        }
        eval_.author = author;
        await eval_.save();
        // NOTE: Side effect. If user email is not set, set it to the author's email
        if (!(0, accounts_1.getUserEmail)()) {
            (0, accounts_1.setUserEmail)(author);
        }
        res.json(apiSchemas_1.ApiSchemas.Eval.UpdateAuthor.Response.parse({
            message: 'Author updated successfully',
        }));
    }
    catch (error) {
        if (error instanceof zod_1.z.ZodError) {
            const validationError = (0, zod_validation_error_1.fromZodError)(error);
            res.status(400).json({ error: validationError.message });
        }
        else {
            logger_1.default.error(`Failed to update eval author: ${error}`);
            res.status(500).json({ error: 'Failed to update eval author' });
        }
    }
});
exports.evalRouter.get('/:id/table', async (req, res) => {
    const { id } = req.params;
    const limit = Number(req.query.limit) || 50;
    const offset = Number(req.query.offset) || 0;
    const filter = String(req.query.filter || 'all');
    const searchText = req.query.search ? String(req.query.search) : '';
    const metricFilter = req.query.metric ? String(req.query.metric) : '';
    const comparisonEvalIds = Array.isArray(req.query.comparisonEvalIds)
        ? req.query.comparisonEvalIds
        : typeof req.query.comparisonEvalIds === 'string'
            ? [req.query.comparisonEvalIds]
            : [];
    const eval_ = await eval_1.default.findById(id);
    if (!eval_) {
        res.status(404).json({ error: 'Eval not found' });
        return;
    }
    const table = await eval_.getTablePage({
        offset,
        limit,
        filterMode: filter,
        searchQuery: searchText,
        metricFilter,
    });
    const indices = table.body.map((row) => row.testIdx);
    let returnTable = { head: table.head, body: table.body };
    if (comparisonEvalIds.length > 0) {
        console.log('comparisonEvalIds', comparisonEvalIds);
        const comparisonEvals = await Promise.all(comparisonEvalIds.map(async (comparisonEvalId) => {
            const comparisonEval_ = await eval_1.default.findById(comparisonEvalId);
            return comparisonEval_;
        }));
        if (comparisonEvals.some((comparisonEval_) => !comparisonEval_)) {
            res.status(404).json({ error: 'Comparison eval not found' });
            return;
        }
        const comparisonTables = await Promise.all(comparisonEvals.map(async (comparisonEval_) => {
            (0, invariant_1.default)(comparisonEval_, 'Comparison eval not found');
            return comparisonEval_.getTablePage({
                offset: 0,
                limit: indices.length,
                filterMode: 'all',
                testIndices: indices,
                searchQuery: searchText,
                metricFilter,
            });
        }));
        returnTable = {
            head: {
                prompts: [
                    ...table.head.prompts.map((prompt) => ({
                        ...prompt,
                        label: `[${id}] ${prompt.label || ''}`,
                    })),
                    ...comparisonTables.flatMap((table) => table.head.prompts.map((prompt) => ({
                        ...prompt,
                        label: `[${table.id}] ${prompt.label || ''}`,
                    }))),
                ],
                vars: table.head.vars, // Assuming vars are the same
            },
            body: table.body.map((row, index) => {
                // Find matching row in comparison table by test index
                const testIdx = row.testIdx;
                const matchingRows = comparisonTables
                    .map((table) => {
                    const compRow = table.body.find((compRow) => {
                        const compTestIdx = compRow.testIdx;
                        return compTestIdx === testIdx;
                    });
                    return compRow;
                })
                    .filter((r) => r !== undefined);
                return {
                    ...row,
                    outputs: [...row.outputs, ...(matchingRows.flatMap((r) => r?.outputs) || [])],
                };
            }),
        };
    }
    res.json({
        table: returnTable,
        totalCount: table.totalCount,
        filteredCount: table.filteredCount,
        config: eval_.config,
        author: eval_.author || null,
        version: eval_.version(),
    });
});
exports.evalRouter.post('/:id/results', async (req, res) => {
    const { id } = req.params;
    const results = req.body;
    if (!Array.isArray(results)) {
        res.status(400).json({ error: 'Results must be an array' });
        return;
    }
    const eval_ = await eval_1.default.findById(id);
    if (!eval_) {
        res.status(404).json({ error: 'Eval not found' });
        return;
    }
    try {
        await eval_.setResults(results);
    }
    catch (error) {
        logger_1.default.error(`Failed to add results to eval: ${error}`);
        res.status(500).json({ error: 'Failed to add results to eval' });
        return;
    }
    res.status(204).send();
});
exports.evalRouter.post('/:evalId/results/:id/rating', async (req, res) => {
    const { id } = req.params;
    const gradingResult = req.body;
    const result = await evalResult_1.default.findById(id);
    (0, invariant_1.default)(result, 'Result not found');
    const eval_ = await eval_1.default.findById(result.evalId);
    (0, invariant_1.default)(eval_, 'Eval not found');
    // Capture the current state before we change it
    const hasExistingManualOverride = Boolean(result.gradingResult?.componentResults?.some((r) => r.assertion?.type === 'human'));
    const successChanged = result.success !== gradingResult.pass;
    const scoreChange = gradingResult.score - result.score;
    // Update the result
    result.gradingResult = gradingResult;
    result.success = gradingResult.pass;
    result.score = gradingResult.score;
    // Update the prompt metrics
    const prompt = eval_.prompts[result.promptIdx];
    (0, invariant_1.default)(prompt, 'Prompt not found');
    if (!prompt.metrics) {
        logger_1.default.error(`[${id}] This is not normal. Prompt metrics not found for prompt ${result.promptIdx}`);
        res.status(400).json({ error: 'Prompt metrics not found' });
        return;
    }
    if (successChanged) {
        if (result.success) {
            // Result changed from fail to pass
            prompt.metrics.testPassCount += 1;
            prompt.metrics.testFailCount -= 1;
            prompt.metrics.assertPassCount += 1;
            prompt.metrics.score += scoreChange;
            if (hasExistingManualOverride) {
                // If there was an existing manual override, we need to decrement the assertFailCount because it changed from fail to pass
                prompt.metrics.assertFailCount -= 1;
            }
        }
        else {
            prompt.metrics.testPassCount -= 1;
            prompt.metrics.testFailCount += 1;
            prompt.metrics.assertFailCount += 1;
            prompt.metrics.score += scoreChange;
            if (hasExistingManualOverride) {
                // If there was an existing manual override, we need to decrement the assertPassCount because it changed from pass to fail
                prompt.metrics.assertPassCount -= 1;
            }
        }
    }
    else if (!hasExistingManualOverride) {
        // Nothing changed, so the user just added an assertion
        if (result.success) {
            prompt.metrics.assertPassCount += 1;
        }
        else {
            prompt.metrics.assertFailCount += 1;
        }
    }
    await eval_.save();
    await result.save();
    res.json(result);
});
exports.evalRouter.post('/', async (req, res) => {
    const body = req.body;
    try {
        if (body.data) {
            logger_1.default.debug('[POST /api/eval] Saving eval results (v3) to database');
            const { data: payload } = req.body;
            const id = await (0, database_1.writeResultsToDatabase)(payload.results, payload.config);
            res.json({ id });
        }
        else {
            const incEval = body;
            logger_1.default.debug('[POST /api/eval] Saving eval results (v4) to database');
            const eval_ = await eval_1.default.create(incEval.config, incEval.prompts || [], {
                author: incEval.author,
                createdAt: new Date(incEval.createdAt),
                results: incEval.results,
            });
            if (incEval.prompts) {
                eval_.addPrompts(incEval.prompts);
            }
            logger_1.default.debug(`[POST /api/eval] Eval created with ID: ${eval_.id}`);
            logger_1.default.debug(`[POST /api/eval] Saved ${incEval.results.length} results to eval ${eval_.id}`);
            res.json({ id: eval_.id });
        }
    }
    catch (error) {
        logger_1.default.error((0, dedent_1.default) `Failed to write eval to database:
      Error: ${error}
      Body: ${JSON.stringify(body, null, 2)}`);
        res.status(500).json({ error: 'Failed to write eval to database' });
    }
});
exports.evalRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        await (0, database_1.deleteEval)(id);
        res.json({ message: 'Eval deleted successfully' });
    }
    catch {
        res.status(500).json({ error: 'Failed to delete eval' });
    }
});
//# sourceMappingURL=eval.js.map