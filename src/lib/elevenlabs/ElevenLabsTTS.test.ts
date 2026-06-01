import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { ElevenLabsTTS, ElevenLabsTTSError } from './ElevenLabsTTS';

/** Build a fake streaming Response that yields the given byte chunks. */
function streamResponse(
  chunks: Uint8Array[],
  { ok = true, status = 200, errorText = 'boom', noBody = false } = {},
) {
  let i = 0;
  return {
    ok,
    status,
    text: async () => errorText,
    body: noBody
      ? null
      : {
          getReader: () => ({
            read: async () =>
              i < chunks.length
                ? { done: false, value: chunks[i++] }
                : { done: true, value: undefined },
            releaseLock: () => {},
          }),
        },
  } as unknown as Response;
}

describe('ElevenLabsTTS', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('frames a single chunk into little-endian Int16 samples', async () => {
    // bytes [0x01,0x00, 0x02,0x00] => Int16 [1, 2]
    fetchMock.mockResolvedValue(
      streamResponse([new Uint8Array([0x01, 0x00, 0x02, 0x00])]),
    );
    const tts = new ElevenLabsTTS({ apiKey: 'k', voiceId: 'v' });

    const out: number[] = [];
    await tts.synthesize('hello', (pcm) => out.push(...pcm));

    expect(out).toEqual([1, 2]);
  });

  it('carries an odd trailing byte across chunk boundaries', async () => {
    // chunk1: [1,0, 2] (one full frame + dangling byte)
    // chunk2: [0, 3,0]  => carried byte completes frame "2", then frame "3"
    fetchMock.mockResolvedValue(
      streamResponse([
        new Uint8Array([0x01, 0x00, 0x02]),
        new Uint8Array([0x00, 0x03, 0x00]),
      ]),
    );
    const tts = new ElevenLabsTTS({ apiKey: 'k', voiceId: 'v' });

    const out: number[] = [];
    await tts.synthesize('hello', (pcm) => out.push(...pcm));

    expect(out).toEqual([1, 2, 3]);
  });

  it('builds the correct streaming URL, headers and body', async () => {
    fetchMock.mockResolvedValue(streamResponse([new Uint8Array([0x00, 0x00])]));
    const tts = new ElevenLabsTTS({
      apiKey: 'secret',
      voiceId: 'voice 1',
      modelId: 'eleven_turbo_v2_5',
    });

    await tts.synthesize('translate me', () => {});

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      'https://api.elevenlabs.io/v1/text-to-speech/voice%201/stream?output_format=pcm_24000',
    );
    expect(init.method).toBe('POST');
    expect(init.headers['xi-api-key']).toBe('secret');
    expect(JSON.parse(init.body)).toEqual({
      text: 'translate me',
      model_id: 'eleven_turbo_v2_5',
    });
    expect('signal' in init).toBe(true);
  });

  it('defaults to the flash model when none is given', async () => {
    fetchMock.mockResolvedValue(streamResponse([new Uint8Array([0x00, 0x00])]));
    const tts = new ElevenLabsTTS({ apiKey: 'k', voiceId: 'v' });

    await tts.synthesize('x', () => {});

    expect(JSON.parse(fetchMock.mock.calls[0][1].body).model_id).toBe(
      'eleven_flash_v2_5',
    );
  });

  it('does not call fetch for empty/whitespace text', async () => {
    const tts = new ElevenLabsTTS({ apiKey: 'k', voiceId: 'v' });
    await tts.synthesize('   ', () => {});
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('throws ElevenLabsTTSError with status on non-ok response', async () => {
    fetchMock.mockResolvedValue(
      streamResponse([], { ok: false, status: 401, errorText: 'unauthorized' }),
    );
    const tts = new ElevenLabsTTS({ apiKey: 'bad', voiceId: 'v' });

    await expect(tts.synthesize('hi', () => {})).rejects.toMatchObject({
      name: 'ElevenLabsTTSError',
      status: 401,
    });
  });

  it('throws when the response has no body stream', async () => {
    fetchMock.mockResolvedValue(streamResponse([], { noBody: true }));
    const tts = new ElevenLabsTTS({ apiKey: 'k', voiceId: 'v' });

    await expect(tts.synthesize('hi', () => {})).rejects.toBeInstanceOf(
      ElevenLabsTTSError,
    );
  });
});
