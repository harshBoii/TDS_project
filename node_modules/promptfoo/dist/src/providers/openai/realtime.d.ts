import WebSocket from 'ws';
import { OpenAiGenericProvider } from '.';
import type { CallApiContextParams, CallApiOptionsParams, ProviderResponse, TokenUsage } from '../../types';
import type { EnvOverrides } from '../../types/env';
import type { OpenAiCompletionOptions } from './types';
export { OPENAI_REALTIME_MODELS } from './util';
export interface OpenAiRealtimeOptions extends OpenAiCompletionOptions {
    modalities?: string[];
    instructions?: string;
    input_audio_format?: 'pcm16' | 'g711_ulaw' | 'g711_alaw';
    input_audio_transcription?: {
        model?: string;
        language?: string;
        prompt?: string;
    } | null;
    output_audio_format?: 'pcm16' | 'g711_ulaw' | 'g711_alaw';
    turn_detection?: {
        type: 'server_vad';
        threshold?: number;
        prefix_padding_ms?: number;
        silence_duration_ms?: number;
        create_response?: boolean;
    } | null;
    voice?: 'alloy' | 'ash' | 'ballad' | 'coral' | 'echo' | 'sage' | 'shimmer' | 'verse';
    max_response_output_tokens?: number | 'inf';
    websocketTimeout?: number;
    tools?: any[];
    tool_choice?: 'none' | 'auto' | 'required' | {
        type: 'function';
        function?: {
            name: string;
        };
    };
    functionCallHandler?: (name: string, args: string) => Promise<string>;
    apiVersion?: string;
    maintainContext?: boolean;
}
interface RealtimeResponse {
    output: string;
    tokenUsage: TokenUsage;
    cached: boolean;
    metadata: any;
    functionCallOccurred?: boolean;
    functionCallResults?: string[];
}
export declare class OpenAiRealtimeProvider extends OpenAiGenericProvider {
    static OPENAI_REALTIME_MODELS: {
        id: string;
        type: string;
        cost: {
            input: number;
            output: number;
            audioInput: number;
            audioOutput: number;
        };
    }[];
    static OPENAI_REALTIME_MODEL_NAMES: string[];
    config: OpenAiRealtimeOptions;
    persistentConnection: WebSocket | null;
    previousItemId: string | null;
    assistantMessageIds: string[];
    private activeTimeouts;
    private lastAudioItemId;
    private currentAudioBuffer;
    private currentAudioFormat;
    private isProcessingAudio;
    private audioTimeout;
    constructor(modelName: string, options?: {
        config?: OpenAiRealtimeOptions;
        id?: string;
        env?: EnvOverrides;
    });
    private resetAudioState;
    getRealtimeSessionBody(): any;
    generateEventId(): string;
    webSocketRequest(clientSecret: string, prompt: string): Promise<RealtimeResponse>;
    callApi(prompt: string, context?: CallApiContextParams, callApiOptions?: CallApiOptionsParams): Promise<ProviderResponse>;
    directWebSocketRequest(prompt: string): Promise<RealtimeResponse>;
    persistentWebSocketRequest(prompt: string): Promise<RealtimeResponse>;
    private setupMessageHandlers;
    cleanup(): void;
}
//# sourceMappingURL=realtime.d.ts.map