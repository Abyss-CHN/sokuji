import { describe, it, expect, beforeEach, vi } from 'vitest';

// Shared mock state, hoisted so the vi.mock factories below can close over it.
const h = vi.hoisted(() => ({
  lastInnerHandlers: null as any,
  lastConnectConfig: null as any,
  innerDisconnect: vi.fn(async () => {}),
  innerReset: vi.fn(),
  ttsConstructArgs: [] as any[],
  synthesize: vi.fn(async (_text: string, onAudio: (pcm: Int16Array) => void) => {
    onAudio(new Int16Array([1, 2, 3]));
  }),
}));

vi.mock('./GeminiClient', () => ({
  GeminiClient: class {
    constructor(public apiKey: string) {}
    setEventHandlers(handlers: any) { h.lastInnerHandlers = handlers; }
    async connect(config: any) { h.lastConnectConfig = config; }
    disconnect = h.innerDisconnect;
    isConnected() { return false; }
    updateSession() {}
    reset = h.innerReset;
    appendInputAudio() {}
    appendInputText() {}
    createResponse() {}
    cancelResponse() {}
    cancelPttTurn() {}
    getConversationItems() { return []; }
    clearConversationItems() {}
    getProvider() { return 'gemini'; }
  },
}));

vi.mock('../../lib/elevenlabs/ElevenLabsTTS', () => ({
  ElevenLabsTTS: class {
    constructor(opts: any) { h.ttsConstructArgs.push(opts); }
    synthesize = h.synthesize;
  },
}));

import { GeminiElevenLabsClient } from './GeminiElevenLabsClient';

const baseConfig: any = {
  provider: 'gemini',
  model: 'gemini-live',
  voice: 'Aoede',
  instructions: 'translate to japanese',
  temperature: 0.8,
  maxTokens: 'inf',
  turnDetectionMode: 'Auto',
  vadStartSensitivity: 'low',
  vadEndSensitivity: 'high',
  vadSilenceDurationMs: 500,
  vadPrefixPaddingMs: 300,
};

const elevenConfig: any = {
  ...baseConfig,
  audioOutputEngine: 'elevenlabs',
  elevenLabsApiKey: 'el-key',
  elevenLabsVoiceId: 'el-voice',
  elevenLabsModelId: 'eleven_flash_v2_5',
  targetLanguage: 'en-US',
};

function assistantItem(id: string, transcript: string, status = 'in_progress') {
  return { id, role: 'assistant', type: 'message', status, formatted: { transcript } };
}

async function waitFor(predicate: () => boolean, timeout = 1000): Promise<void> {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > timeout) throw new Error('waitFor: condition not met');
    await new Promise((r) => setTimeout(r, 5));
  }
}

function makeClient() {
  const outer = {
    onOpen: vi.fn(),
    onClose: vi.fn(),
    onError: vi.fn(),
    onConversationUpdated: vi.fn(),
    onConversationInterrupted: vi.fn(),
  };
  const client = new GeminiElevenLabsClient('gemini-key');
  client.setEventHandlers(outer);
  return { client, outer };
}

function audioDeltaCalls(outer: any) {
  return outer.onConversationUpdated.mock.calls.filter((c: any[]) => c[0]?.delta?.audio);
}

describe('GeminiElevenLabsClient', () => {
  beforeEach(() => {
    h.lastInnerHandlers = null;
    h.lastConnectConfig = null;
    h.ttsConstructArgs.length = 0;
    h.innerDisconnect.mockClear();
    h.innerReset.mockClear();
    h.synthesize.mockClear();
    h.synthesize.mockImplementation(async (_text: string, onAudio: (pcm: Int16Array) => void) => {
      onAudio(new Int16Array([1, 2, 3]));
    });
  });

  it('reports the Gemini provider', () => {
    const { client } = makeClient();
    expect(client.getProvider()).toBe('gemini');
  });

  it('forwards lifecycle events from the inner client', async () => {
    const { outer } = makeClient();
    h.lastInnerHandlers.onOpen();
    h.lastInnerHandlers.onError(new Error('x'));
    expect(outer.onOpen).toHaveBeenCalledTimes(1);
    expect(outer.onError).toHaveBeenCalledTimes(1);
  });

  describe('passthrough (engine = gemini)', () => {
    it('does not force textOnly and never synthesizes', async () => {
      const { client, outer } = makeClient();
      await client.connect({ ...baseConfig, audioOutputEngine: 'gemini' });

      expect(h.lastConnectConfig.textOnly).toBeUndefined();
      expect(h.ttsConstructArgs).toHaveLength(0);

      h.lastInnerHandlers.onConversationUpdated({
        item: assistantItem('a1', 'Hello world.', 'completed'),
      });
      await new Promise((r) => setTimeout(r, 10));

      expect(h.synthesize).not.toHaveBeenCalled();
      // The transcript update is still forwarded to the UI.
      expect(outer.onConversationUpdated).toHaveBeenCalledTimes(1);
    });
  });

  describe('ElevenLabs engine', () => {
    it('forces Gemini into text-only mode and constructs the TTS engine', async () => {
      const { client } = makeClient();
      await client.connect(elevenConfig);

      expect(h.lastConnectConfig.textOnly).toBe(true);
      expect(h.ttsConstructArgs[0]).toMatchObject({
        apiKey: 'el-key',
        voiceId: 'el-voice',
        modelId: 'eleven_flash_v2_5',
      });
    });

    it('rejects connect when API key or voice is missing', async () => {
      const { client } = makeClient();
      await expect(
        client.connect({ ...elevenConfig, elevenLabsApiKey: '', elevenLabsVoiceId: '' }),
      ).rejects.toThrow(/ElevenLabs/);
    });

    it('synthesizes completed sentences incrementally, flushing the last on turn end', async () => {
      const { client, outer } = makeClient();
      await client.connect(elevenConfig);

      // Streaming: first sentence complete, second still in progress.
      h.lastInnerHandlers.onConversationUpdated({ item: assistantItem('a1', 'Hello world. How') });
      // More of the second sentence — still no new completed sentence.
      h.lastInnerHandlers.onConversationUpdated({ item: assistantItem('a1', 'Hello world. How are you?') });
      // Turn completes — the final sentence flushes.
      h.lastInnerHandlers.onConversationUpdated({ item: assistantItem('a1', 'Hello world. How are you?', 'completed') });

      await waitFor(() => h.synthesize.mock.calls.length === 2);
      expect(h.synthesize.mock.calls[0][0]).toBe('Hello world.');
      expect(h.synthesize.mock.calls[1][0]).toBe('How are you?');

      // Each synthesized sentence produced an audio delta routed to the player,
      // tagged with the originating assistant item.
      const deltas = audioDeltaCalls(outer);
      expect(deltas).toHaveLength(2);
      expect(deltas[0][0].item.id).toBe('a1');
      expect(Array.from(deltas[0][0].delta.audio)).toEqual([1, 2, 3]);
    });

    it('does not re-synthesize sentences already dispatched', async () => {
      const { client } = makeClient();
      await client.connect(elevenConfig);

      h.lastInnerHandlers.onConversationUpdated({ item: assistantItem('a1', 'One. Two.') });
      await waitFor(() => h.synthesize.mock.calls.length === 1);
      // Same transcript arrives again (e.g. a duplicate finalize) — no new work
      // for the already-spoken first sentence.
      h.lastInnerHandlers.onConversationUpdated({ item: assistantItem('a1', 'One. Two.') });
      await new Promise((r) => setTimeout(r, 10));

      expect(h.synthesize.mock.calls.map((c) => c[0])).toEqual(['One.']);
    });

    it('aborts and clears queued TTS on interruption', async () => {
      const { client, outer } = makeClient();
      await client.connect(elevenConfig);

      h.lastInnerHandlers.onConversationUpdated({ item: assistantItem('a1', 'Hi there.', 'completed') });
      await waitFor(() => h.synthesize.mock.calls.length === 1);

      h.lastInnerHandlers.onConversationInterrupted();
      expect(outer.onConversationInterrupted).toHaveBeenCalledTimes(1);

      // A brand-new turn after interruption synthesizes from scratch.
      h.lastInnerHandlers.onConversationUpdated({ item: assistantItem('a2', 'New turn.', 'completed') });
      await waitFor(() => h.synthesize.mock.calls.length === 2);
      expect(h.synthesize.mock.calls[1][0]).toBe('New turn.');
    });

    it('disconnect tears down TTS and the inner client', async () => {
      const { client } = makeClient();
      await client.connect(elevenConfig);
      await client.disconnect();
      expect(h.innerDisconnect).toHaveBeenCalledTimes(1);
    });
  });
});
