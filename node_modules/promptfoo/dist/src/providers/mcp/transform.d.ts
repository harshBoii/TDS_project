import type Anthropic from '@anthropic-ai/sdk';
import type { Tool as GoogleTool } from '../google/types';
import type { OpenAiTool } from '../openai/util';
import type { MCPTool } from './types';
export declare function transformMCPToolsToOpenAi(tools: MCPTool[]): OpenAiTool[];
export declare function transformMCPToolsToAnthropic(tools: MCPTool[]): Anthropic.Tool[];
export declare function transformMCPToolsToGoogle(tools: MCPTool[]): GoogleTool[];
//# sourceMappingURL=transform.d.ts.map