/**
 * ElevenLabs streaming text-to-speech engine.
 *
 * Used by GeminiElevenLabsClient to turn translated text (produced by Gemini in
 * text-only mode) into speech. We request the `pcm_24000` output format, which
 * is raw 16-bit signed little-endian mono PCM at 24 kHz — exactly what
 * ModernAudioPlayer consumes (see the Gemini/LocalInference audio paths), so no
 * resampling is needed: incoming bytes are framed into Int16Array and emitted.
 *
 * The HTTP streaming endpoint (`/stream`) starts returning audio bytes before
 * synthesis finishes, so per-sentence calls keep time-to-first-audio low.
 */

export interface ElevenLabsTTSOptions {
  apiKey: string;
  voiceId: string;
  /** Default: 'eleven_flash_v2_5' — multilingual, low latency (~75ms). */
  modelId?: string;
  /**
   * ElevenLabs output_format query value. Must be a raw PCM format so the bytes
   * map directly onto Int16Array. Default: 'pcm_24000' (24 kHz, matches the
   * player's native rate). Other pcm_* rates would require resampling upstream.
   */
  outputFormat?: string;
  /** Override for tests; defaults to the public ElevenLabs API host. */
  baseUrl?: string;
}

export const DEFAULT_ELEVENLABS_MODEL = 'eleven_flash_v2_5';
export const DEFAULT_ELEVENLABS_OUTPUT_FORMAT = 'pcm_24000';
const DEFAULT_BASE_URL = 'https://api.elevenlabs.io';

export class ElevenLabsTTSError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message);
    this.name = 'ElevenLabsTTSError';
  }
}

export class ElevenLabsTTS {
  private readonly apiKey: string;
  private readonly voiceId: string;
  private readonly modelId: string;
  private readonly outputFormat: string;
  private readonly baseUrl: string;

  constructor(options: ElevenLabsTTSOptions) {
    this.apiKey = options.apiKey;
    this.voiceId = options.voiceId;
    this.modelId = options.modelId || DEFAULT_ELEVENLABS_MODEL;
    this.outputFormat = options.outputFormat || DEFAULT_ELEVENLABS_OUTPUT_FORMAT;
    this.baseUrl = (options.baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, '');
  }

  /**
   * Stream-synthesize `text`, invoking `onAudio` with Int16Array PCM chunks as
   * they arrive (in order). Resolves when the full response has been consumed.
   *
   * Aborting `signal` cancels the in-flight request; the resulting AbortError is
   * re-thrown so callers can distinguish cancellation from genuine failures.
   */
  async synthesize(
    text: string,
    onAudio: (pcm: Int16Array) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed) return;

    const url =
      `${this.baseUrl}/v1/text-to-speech/${encodeURIComponent(this.voiceId)}/stream` +
      `?output_format=${encodeURIComponent(this.outputFormat)}`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'xi-api-key': this.apiKey,
        'Content-Type': 'application/json',
        Accept: 'audio/pcm',
      },
      body: JSON.stringify({
        text: trimmed,
        model_id: this.modelId,
      }),
      signal,
    });

    if (!response.ok) {
      // Surface the server's error text (quota/auth/voice issues) when available.
      const detail = await response.text().catch(() => '');
      throw new ElevenLabsTTSError(
        `ElevenLabs TTS request failed (${response.status})${detail ? `: ${detail}` : ''}`,
        response.status,
      );
    }
    if (!response.body) {
      throw new ElevenLabsTTSError('ElevenLabs TTS response has no body stream');
    }

    const reader = response.body.getReader();
    // Holds at most one trailing byte that did not complete a 16-bit frame in
    // the previous chunk, prepended to the next chunk.
    let carry: Uint8Array | null = null;

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value || value.length === 0) continue;

        let bytes: Uint8Array = value;
        if (carry) {
          const merged = new Uint8Array(carry.length + value.length);
          merged.set(carry, 0);
          merged.set(value, carry.length);
          bytes = merged;
          carry = null;
        }

        const usableLength = bytes.length - (bytes.length % 2);
        if (usableLength > 0) {
          // Copy into a fresh, 2-byte-aligned buffer so Int16Array can wrap it
          // directly. `slice` yields a buffer with byteOffset 0.
          const frame = bytes.slice(0, usableLength);
          onAudio(new Int16Array(frame.buffer, 0, usableLength / 2));
        }
        if (usableLength < bytes.length) {
          carry = bytes.slice(usableLength);
        }
      }
    } finally {
      reader.releaseLock?.();
    }
  }
}
