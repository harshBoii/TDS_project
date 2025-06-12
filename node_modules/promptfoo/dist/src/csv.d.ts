import type { Assertion, CsvRow, TestCase } from './types';
export declare function assertionFromString(expected: string): Assertion;
export declare function testCaseFromCsvRow(row: CsvRow): TestCase;
/**
 * Serialize a list of VarMapping objects as a CSV string.
 * @param vars - The list of VarMapping objects to serialize.
 * @returns A CSV string.
 */
export declare function serializeObjectArrayAsCSV(vars: object[]): string;
//# sourceMappingURL=csv.d.ts.map