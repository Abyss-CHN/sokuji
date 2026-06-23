/**
 * GeminiElevenLabsClient — a transparent decorator over GeminiClient that
 * replaces Gemini's native voice output with ElevenLabs text-to-speech while
 * keeping Gemini as the translation engine.
 *
 * Behaviour is selected per-session from `GeminiSessionConfig.audioOutputEngine`:
 *
 *  - `'gemini'` (or unset): pure pass-through. Every IClient call is forwarded
 *    to the inner GeminiClient unchanged, so the session is byte-for-byte
 *    identical to using GeminiClient directly.
 *
 *  - `'elevenlabs'`: the inner GeminiClient is forced into `textOnly` mode (it
 *    emits the translated transcript but no audio). This decorator watches the
 *    assistant transcript, splits it into sentences as it streams, and feeds
 *    completed sentences to ElevenLabs TTS. The returned PCM is emitted as
 *    `onConversationUpdated({ delta: { audio } })` — the same channel Gemini's
 *    native audio uses — so MainPanel/ModernAudioPlayer play it transparently.
 *
 * The factory always wraps Gemini in this decorator; the transparent path keeps
 * that free of behavioural change. API-key validation and model listing are
 * static methods invoked on GeminiClient directly (via ClientOperations), so
 * they are unaffected by this wrapper.
 */

import {
  IClient,
  ConversationItem,
  SessionConfig,
  ClientEventHandlers,
  ResponseConfig,
  isGeminiSessionConfig,
} from '../interfaces/IClient';
import { Provider, ProviderType } from '../../types/Provider';
import { splitSentences } from '../../utils/splitSentences';
import i18n from '../../locales';
import { ElevenLabsTTS, ElevenLabsTTSError } from '../../lib/elevenlabs/ElevenLabsTTS';
import { GeminiClient } from './GeminiClient';

interface TtsJob {
  item: ConversationItem;
  text: string;
}

export class GeminiElevenLabsClient implements IClient {
  private readonly inner: GeminiClient;
  private outer: ClientEventHandlers = {};

  // TTS state — only active when ttsEnabled (engine === 'elevenlabs').
  private ttsEnabled = false;
  private tts: ElevenLabsTTS | null = null;
  private targetLocale = 'en';
  /** Per assistant item id: how many leading sentences have been sent to TTS. */
  private readonly dispatchedSentences = new Map<string, number>();
  private readonly ttsQueue: TtsJob[] = [];
  private ttsProcessing = false;
  private ttsAbort: AbortController | null = null;
  /**
   * Set after a non-transient TTS failure (bad key / plan / voice). Stops
   * further synthesis for the rest of the session so we don't hammer the API
   * (or spam errors) on every subsequent sentence.
   */
  private ttsFatal = false;
  /** Ensures only ONE user-facing error is surfaced per session. */
  private ttsErrorNotified = false;

  constructor(apiKey: string) {
    this.inner = new GeminiClient(apiKey);

    // Register inner handlers ONCE. They dispatch through `this.outer`
    // dynamically, so the decorator's own setEventHandlers only needs to swap
    // that reference — no re-registration races regardless of call order.
    this.inner.setEventHandlers({
      onOpen: () => this.outer.onOpen?.(),
      onClose: (event) => this.outer.onClose?.(event),
      onError: (error) => this.outer.onError?.(error),
      onRealtimeEvent: (event) => this.outer.onRealtimeEvent?.(event),
      onReconnecting: () => this.outer.onReconnecting?.(),
      onReconnected: () => this.outer.onReconnected?.(),
      onConversationInterrupted: () => {
        if (this.ttsEnabled) this.abortTts();
        this.outer.onConversationInterrupted?.();
      },
      onConversationUpdated: (data) => {
        // Forward the original event verbatim so the UI text path is unchanged.
        this.outer.onConversationUpdated?.(data);
        // Drive TTS off the assistant transcript (skip our own audio deltas,
        // which never carry a fresh transcript to re-synthesize).
        if (this.ttsEnabled && !data.delta?.audio) {
          this.maybeSynthesize(data.item);
        }
      },
    });
  }

  async connect(config: SessionConfig): Promise<void> {
    this.resetTtsState();

    this.ttsEnabled =
      isGeminiSessionConfig(config) && config.audioOutputEngine === 'elevenlabs';

    if (this.ttsEnabled && isGeminiSessionConfig(config)) {
      const apiKey = config.elevenLabsApiKey?.trim();
      const voiceId = config.elevenLabsVoiceId?.trim();
      if (!apiKey || !voiceId) {
        throw new Error(
          'ElevenLabs API key and voice are required when the audio output engine is ElevenLabs',
        );
      }
      this.tts = new ElevenLabsTTS({
        apiKey,
        voiceId,
        modelId: config.elevenLabsModelId,
      });
      this.targetLocale = config.targetLanguage || 'en';

      // Force Gemini to text-only so it produces the transcript but no audio —
      // ElevenLabs is the sole audio source in this mode.
      return this.inner.connect({ ...config, textOnly: true });
    }

    this.tts = null;
    return this.inner.connect(config);
  }

  /**
   * Inspect an assistant item's transcript and dispatch any newly-completed
   * sentences to the TTS queue. Idempotent: a per-item counter ensures each
   * sentence is synthesized exactly once even though the item updates many
   * times as it streams.
   */
  private maybeSynthesize(item: ConversationItem): void {
    if (this.ttsFatal) return; // a prior non-transient failure disabled TTS this session
    if (!item || item.role !== 'assistant' || item.type !== 'message') return;

    // Synthesize ONLY the spoken-output transcription, never formatted.text.
    // For Gemini native-audio models, `transcript` (outputAudioTranscription)
    // is the clean translation the model actually speaks, whereas `text`
    // (modelTurn text parts) can carry the model's thinking / preamble. Those
    // text parts often arrive BEFORE the transcript, so falling back to them
    // made ElevenLabs read the "extra prompt words" aloud — native Gemini
    // never voices them. Transcript-only keeps TTS aligned with native speech.
    const text = item.formatted?.transcript ?? '';
    if (!text.trim()) return;

    const sentences = splitSentences(text, this.targetLocale);
    // While streaming, the final sentence may still be growing, so only treat
    // it as complete once the turn finishes.
    const completeCount =
      item.status === 'completed' ? sentences.length : Math.max(0, sentences.length - 1);

    const already = this.dispatchedSentences.get(item.id) ?? 0;
    if (completeCount <= already) return;

    const pending = sentences.slice(already, completeCount);
    this.dispatchedSentences.set(item.id, completeCount);
    for (const sentence of pending) {
      this.ttsQueue.push({ item, text: sentence });
    }
    void this.processTtsQueue();
  }

  private async processTtsQueue(): Promise<void> {
    if (this.ttsProcessing || !this.tts) return;
    this.ttsProcessing = true;

    try {
      while (this.ttsQueue.length > 0) {
        const job = this.ttsQueue.shift()!;
        const abort = new AbortController();
        this.ttsAbort = abort;
        try {
          await this.tts.synthesize(
            job.text,
            (pcm) => {
              // Re-check: a barge-in interrupt may have fired mid-stream.
              if (abort.signal.aborted) return;
              this.outer.onConversationUpdated?.({
                item: job.item,
                delta: { audio: pcm },
              });
            },
            abort.signal,
          );
        } catch (error) {
          if (!abort.signal.aborted) {
            this.handleTtsError(error);
          }
          // Aborted jobs fall through; the queue was cleared by abortTts().
        } finally {
          if (this.ttsAbort === abort) this.ttsAbort = null;
        }
      }
    } finally {
      this.ttsProcessing = false;
    }
  }

  /** Cancel in-flight + queued TTS (barge-in / interruption / teardown). */
  private abortTts(): void {
    this.ttsAbort?.abort();
    this.ttsAbort = null;
    this.ttsQueue.length = 0;
    this.dispatchedSentences.clear();
  }

  private resetTtsState(): void {
    this.abortTts();
    this.ttsProcessing = false;
    this.ttsFatal = false;
    this.ttsErrorNotified = false;
  }

  /**
   * Handle a TTS request failure: always log to the LogsPanel; surface ONE
   * clear, actionable message to the user (via onError → MainPanel shows it in
   * the conversation panel); and for non-transient failures, disable TTS for
   * the rest of the session so we don't spam the API/logs on every sentence.
   */
  private handleTtsError(error: unknown): void {
    const status = error instanceof ElevenLabsTTSError ? error.status : undefined;
    const detail = error instanceof Error ? error.message : String(error);
    console.error('[Sokuji] [GeminiElevenLabsClient] TTS error:', error);

    // Diagnostics — every failure, for the LogsPanel.
    this.outer.onRealtimeEvent?.({
      source: 'client',
      event: { type: 'elevenlabs.tts.error', data: { status, message: detail } },
    });

    // Bad key / plan / voice / request won't recover mid-session — stop trying.
    const fatal = status !== undefined && [400, 401, 402, 403, 404].includes(status);
    if (fatal) {
      this.ttsFatal = true;
      this.ttsQueue.length = 0;
    }

    // One user-facing notification per session (every sentence would otherwise
    // re-fire the same error).
    if (!this.ttsErrorNotified) {
      this.ttsErrorNotified = true;
      this.outer.onError?.({ message: this.userFacingTtsMessage(status, detail) });
    }
  }

  /** Map an ElevenLabs failure to a clear, actionable, localized message. */
  private userFacingTtsMessage(status: number | undefined, detail: string): string {
    switch (status) {
      case 401:
        return i18n.t('errors.elevenLabsTts.invalidKey');
      case 402:
        return i18n.t('errors.elevenLabsTts.planRequired');
      case 403:
        return i18n.t('errors.elevenLabsTts.denied');
      case 404:
        return i18n.t('errors.elevenLabsTts.voiceNotFound');
      case 429:
        return i18n.t('errors.elevenLabsTts.rateLimited');
      default:
        if (status !== undefined && status >= 500) {
          return i18n.t('errors.elevenLabsTts.serverError', { status });
        }
        return status !== undefined
          ? i18n.t('errors.elevenLabsTts.genericWithStatus', { status, detail })
          : i18n.t('errors.elevenLabsTts.generic', { detail });
    }
  }

  // --- IClient pass-through surface ---------------------------------------

  async disconnect(): Promise<void> {
    this.resetTtsState();
    await this.inner.disconnect();
  }

  isConnected(): boolean {
    return this.inner.isConnected();
  }

  updateSession(config: Partial<SessionConfig>): void {
    this.inner.updateSession(config);
  }

  reset(): void {
    this.resetTtsState();
    this.inner.reset();
  }

  appendInputAudio(audioData: Int16Array): void {
    this.inner.appendInputAudio(audioData);
  }

  appendInputText(text: string): void {
    this.inner.appendInputText(text);
  }

  createResponse(config?: ResponseConfig): void {
    this.inner.createResponse(config);
  }

  cancelResponse(trackId?: string, offset?: number): void {
    if (this.ttsEnabled) this.abortTts();
    this.inner.cancelResponse(trackId, offset);
  }

  cancelPttTurn(): void {
    this.inner.cancelPttTurn?.();
  }

  getConversationItems(): ConversationItem[] {
    return this.inner.getConversationItems();
  }

  clearConversationItems(): void {
    this.dispatchedSentences.clear();
    this.inner.clearConversationItems();
  }

  setEventHandlers(handlers: ClientEventHandlers): void {
    this.outer = { ...handlers };
  }

  getProvider(): ProviderType {
    return Provider.GEMINI;
  }
}
