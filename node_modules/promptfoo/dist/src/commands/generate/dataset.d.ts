import type { Command } from 'commander';
import { type UnifiedConfig } from '../../types';
interface DatasetGenerateOptions {
    cache: boolean;
    config?: string;
    envFile?: string;
    instructions?: string;
    numPersonas: string;
    numTestCasesPerPersona: string;
    output?: string;
    provider?: string;
    write: boolean;
    defaultConfig: Partial<UnifiedConfig>;
    defaultConfigPath: string | undefined;
}
export declare function doGenerateDataset(options: DatasetGenerateOptions): Promise<void>;
export declare function generateDatasetCommand(program: Command, defaultConfig: Partial<UnifiedConfig>, defaultConfigPath: string | undefined): void;
export {};
//# sourceMappingURL=dataset.d.ts.map