"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.notCloudEnabledShareInstructions = notCloudEnabledShareInstructions;
exports.createAndDisplayShareableUrl = createAndDisplayShareableUrl;
exports.shareCommand = shareCommand;
const confirm_1 = __importDefault(require("@inquirer/confirm"));
const chalk_1 = __importDefault(require("chalk"));
const dedent_1 = __importDefault(require("dedent"));
const constants_1 = require("../constants");
const cloud_1 = require("../globalConfig/cloud");
const logger_1 = __importDefault(require("../logger"));
const eval_1 = __importDefault(require("../models/eval"));
const share_1 = require("../share");
const telemetry_1 = __importDefault(require("../telemetry"));
const default_1 = require("../util/config/default");
function notCloudEnabledShareInstructions() {
    const cloudUrl = (0, constants_1.getDefaultShareViewBaseUrl)();
    const welcomeUrl = `${cloudUrl}/welcome`;
    logger_1.default.info((0, dedent_1.default) `
    
    » You need to have a cloud account to securely share your results.
    
    1. Please go to ${chalk_1.default.greenBright.bold(cloudUrl)} to sign up or log in.
    2. Follow the instructions at ${chalk_1.default.greenBright.bold(welcomeUrl)} to login to the command line.
    3. Run ${chalk_1.default.greenBright.bold('promptfoo share')}
  `);
}
async function createAndDisplayShareableUrl(evalRecord, showAuth) {
    const url = await (0, share_1.createShareableUrl)(evalRecord, showAuth);
    if (url) {
        logger_1.default.info(`View results: ${chalk_1.default.greenBright.bold(url)}`);
    }
    else {
        logger_1.default.error('Failed to create shareable URL');
        process.exitCode = 1;
    }
    return url;
}
function shareCommand(program) {
    program
        .command('share [evalId]')
        .description('Create a shareable URL of an eval (defaults to most recent)' + '\n\n')
        .option('--show-auth', 'Show username/password authentication information in the URL if exists', false)
        // NOTE: Added in 0.109.1 after migrating sharing to promptfoo.app in 0.108.0
        .option('-y, --yes', 'Flag does nothing (maintained for backwards compatibility only - shares are now private by default)', false)
        .action(async (evalId, cmdObj) => {
        telemetry_1.default.record('command_used', {
            name: 'share',
        });
        await telemetry_1.default.send();
        let eval_ = null;
        if (evalId) {
            eval_ = await eval_1.default.findById(evalId);
            if (!eval_) {
                logger_1.default.error(`Could not find eval with ID ${chalk_1.default.bold(evalId)}.`);
                process.exitCode = 1;
                return;
            }
        }
        else {
            eval_ = await eval_1.default.latest();
            if (!eval_) {
                logger_1.default.error('Could not load results. Do you need to run `promptfoo eval` first?');
                process.exitCode = 1;
                return;
            }
            logger_1.default.info(`Sharing latest eval (${eval_.id})`);
        }
        try {
            const { defaultConfig: currentConfig } = await (0, default_1.loadDefaultConfig)();
            if (currentConfig && currentConfig.sharing) {
                eval_.config.sharing = currentConfig.sharing;
                logger_1.default.debug(`Applied sharing config from promptfooconfig.yaml: ${JSON.stringify(currentConfig.sharing)}`);
            }
        }
        catch (err) {
            logger_1.default.debug(`Could not load config: ${err}`);
        }
        if (eval_.prompts.length === 0) {
            // FIXME(ian): Handle this on the server side.
            logger_1.default.error((0, dedent_1.default) `
              Eval ${chalk_1.default.bold(eval_.id)} cannot be shared.
              This may be because the eval is still running or because it did not complete successfully.
              If your eval is still running, wait for it to complete and try again.
            `);
            process.exitCode = 1;
            return;
        }
        // Validate that the user has authenticated with Cloud.
        if (!(0, share_1.isSharingEnabled)(eval_)) {
            notCloudEnabledShareInstructions();
            process.exitCode = 1;
            return;
        }
        if (
        // Idempotency is not implemented in self-hosted mode.
        cloud_1.cloudConfig.isEnabled() &&
            (await (0, share_1.hasEvalBeenShared)(eval_))) {
            const url = await (0, share_1.getShareableUrl)(eval_, cmdObj.showAuth);
            const shouldContinue = await (0, confirm_1.default)({
                message: `This eval is already shared at ${url}. Sharing it again will overwrite the existing data. Continue?`,
            });
            if (!shouldContinue) {
                process.exitCode = 0;
                return;
            }
        }
        await createAndDisplayShareableUrl(eval_, cmdObj.showAuth);
    });
}
//# sourceMappingURL=share.js.map