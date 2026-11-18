import { beforeAll, describe, expect, test, vi } from 'vitest';
import {
  getVideoSampleTimes,
  sampleVideo,
  sampleVideoFrames,
  videoToMessageContent,
} from './video';
import { Wllama } from './wllama';

const VISION_MODEL = {
  url: 'https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/tinygemma3-Q8_0.gguf',
  mmprojUrl:
    'https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/mmproj-tinygemma3.gguf',
};

// merges each 2 video frames, see MTMD_VIDEO_TEMPLATE_LFM3_OMNI
const MERGE_MODEL_REPO =
  'https://huggingface.co/ggml-org/tinyLFM3-Omni-for-testing-gguf/resolve/main';
const MERGE_MODEL = {
  url: `${MERGE_MODEL_REPO}/tinyLFM3-Omni-for-testing-Q2_K.gguf`,
  mmprojUrl: `${MERGE_MODEL_REPO}/mmproj-tinyLFM3-Omni-for-testing-Q2_K.gguf`,
};

const WIDTH = 640;
const HEIGHT = 320;

// 2 seconds of VP8 at 10 fps: red for the first second, then blue. Only the first frame is a keyframe, so a seek must be exact
// ffmpeg -f lavfi -i color=red:s=640x320:d=1:r=10 -f lavfi -i color=blue:s=640x320:d=1:r=10 -filter_complex "[0][1]concat=n=2:v=1" -c:v libvpx -g 300 -keyint_min 300 -auto-alt-ref 0
const CLIP_BASE64 =
  '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';

const loadClip = (): Blob =>
  new Blob([Uint8Array.from(atob(CLIP_BASE64), (c) => c.charCodeAt(0))], {
    type: 'video/webm',
  });

const readFrame = async (data: ArrayBuffer) => {
  const bitmap = await createImageBitmap(
    new Blob([data], { type: 'image/jpeg' })
  );
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(bitmap, 0, 0);
  const [r, , b] = ctx.getImageData(
    bitmap.width >> 1,
    bitmap.height >> 1,
    1,
    1
  ).data;
  return { width: bitmap.width, height: bitmap.height, r, b };
};

describe('getVideoSampleTimes', () => {
  test('samples at the target fps', () => {
    const times = getVideoSampleTimes(10, 2, 100);
    expect(times).toHaveLength(20);
    expect(times[0]).toBe(0);
    expect(times[1]).toBeCloseTo(0.5);
    expect(times[19]).toBeCloseTo(9.5);
  });

  test('limits to maxFrames and spreads over the whole video', () => {
    const times = getVideoSampleTimes(600, 2, 8);
    expect(times).toHaveLength(8);
    expect(times[7]).toBeCloseTo(525);
  });

  test('uses at least 4 frames unless maxFrames is lower', () => {
    expect(getVideoSampleTimes(0.5, 2, 32)).toHaveLength(4);
    expect(getVideoSampleTimes(0.5, 2, 2)).toHaveLength(2);
  });

  test('handles a single frame and a very short video', () => {
    expect(getVideoSampleTimes(10, 2, 1)).toEqual([0]);
    expect(getVideoSampleTimes(0.04, 2, 32)).toEqual([0, 0.01, 0.02, 0.03]);
  });
});

describe('sampleVideoFrames', () => {
  let clip: Blob;

  beforeAll(() => {
    clip = loadClip();
  });

  test('reads the frames at the correct positions', async () => {
    const frames = await sampleVideoFrames(clip, { maxFrames: 4 });
    expect(frames).toHaveLength(4);
    expect(frames[0].time).toBe(0);

    const first = await readFrame(frames[0].data);
    expect(first.r).toBeGreaterThan(200);
    expect(first.b).toBeLessThan(60);

    const last = await readFrame(frames[3].data);
    expect(last.b).toBeGreaterThan(200);
    expect(last.r).toBeLessThan(60);
  }, 30000);

  test('scales frames down and keeps the aspect ratio', async () => {
    const frames = await sampleVideoFrames(clip, {
      maxFrames: 1,
      maxEdge: 64,
    });
    const { width, height } = await readFrame(frames[0].data);
    expect(width).toBe(64);
    expect(height).toBe(32);
  }, 30000);

  test('does not upscale frames', async () => {
    const frames = await sampleVideoFrames(clip, {
      maxFrames: 1,
      maxEdge: 4096,
    });
    const { width, height } = await readFrame(frames[0].data);
    expect(width).toBe(WIDTH);
    expect(height).toBe(HEIGHT);
  }, 30000);

  test('reads a URL', async () => {
    const url = URL.createObjectURL(clip);
    const frames = await sampleVideoFrames(url, { maxFrames: 1 });
    URL.revokeObjectURL(url);
    expect(frames).toHaveLength(1);
  }, 30000);

  test('seeks to the end if the duration is unknown', async () => {
    const proto = HTMLMediaElement.prototype;
    const getDuration = Object.getOwnPropertyDescriptor(
      proto,
      'duration'
    )!.get!;
    let known = false;
    const spy = vi.spyOn(proto, 'duration', 'get').mockImplementation(function (
      this: HTMLMediaElement
    ) {
      known ||= this.currentTime > 0;
      return known ? getDuration.call(this) : Infinity;
    });
    try {
      const frames = await sampleVideoFrames(clip, { maxFrames: 2 });
      expect(known).toBe(true);
      expect(frames[1].time).toBeGreaterThan(0.5);
    } finally {
      spy.mockRestore();
    }
  }, 30000);

  test('rejects a file that is not a video', async () => {
    await expect(
      sampleVideoFrames(new Blob(['not a video'], { type: 'video/mp4' }))
    ).rejects.toThrow('Cannot read video');
  }, 30000);

  test('rejects invalid options', async () => {
    await expect(sampleVideoFrames(clip, { maxFrames: 0 })).rejects.toThrow(
      'must be positive'
    );
    await expect(sampleVideoFrames(clip, { fps: -1 })).rejects.toThrow(
      'must be positive'
    );
  });

  test('puts a timestamp before each frame', async () => {
    const content = await videoToMessageContent(clip, { maxFrames: 2 });
    expect(content.map((c) => c.type)).toEqual([
      'text',
      'image',
      'text',
      'image',
    ]);
    expect(content[0]).toEqual({ type: 'text', text: '<0.0 seconds>' });
    expect(content[2].type === 'text' && content[2].text).toMatch(
      /^<\d+\.\d seconds>$/
    );
  }, 30000);

  test('packs raw frames for native video input', async () => {
    const data = await sampleVideo(clip, { maxFrames: 4, maxEdge: 64 });
    const header = new DataView(data);
    const [width, height, count] = [8, 12, 16].map((pos) =>
      header.getUint32(pos, true)
    );
    expect(new TextDecoder().decode(data.slice(0, 4))).toBe('WLVF');
    expect(header.getUint32(4, true)).toBe(1);
    expect([width, height, count]).toEqual([64, 32, 4]);
    expect(header.getFloat32(20, true)).toBeGreaterThan(1.5); // 4 frames in about 2 seconds
    expect(header.getFloat32(20, true)).toBeLessThan(2.5);
    expect(data.byteLength).toBe(24 + 4 * 64 * 32 * 3);

    const frameSize = width * height * 3;
    const center = (frame: number) =>
      new Uint8Array(
        data,
        24 + frame * frameSize + ((height >> 1) * width + (width >> 1)) * 3,
        3
      );
    expect(center(0)[0]).toBeGreaterThan(200);
    expect(center(0)[2]).toBeLessThan(60);
    expect(center(3)[2]).toBeGreaterThan(200);
    expect(center(3)[0]).toBeLessThan(60);
  }, 30000);

  test.sequential(
    'sends the frames to a vision model',
    async () => {
      const wllama = new Wllama({ default: '/src/wasm/wllama.wasm' });
      wllama.setCompat(null);
      await wllama.loadModelFromUrl(VISION_MODEL, {
        n_ctx: 4096,
        n_threads: 4,
      });

      const promptTokens = async (maxFrames: number) => {
        const res = await wllama.createChatCompletion({
          messages: [
            {
              role: 'user',
              content: [
                ...(await videoToMessageContent(clip, { maxFrames })),
                { type: 'text', text: 'Describe this video' },
              ],
            },
          ],
          max_tokens: 4,
        });
        expect(res.choices[0].message.content).toBeTypeOf('string');
        return res.usage!.prompt_tokens;
      };

      const one = await promptTokens(1);
      const three = await promptTokens(3);
      expect(three).toBeGreaterThan(one * 2);

      // native video input
      expect(wllama.supportInputModality('video')).toBe(true);
      const nativePromptTokens = async (maxFrames: number) => {
        const res = await wllama.createChatCompletion({
          messages: [
            {
              role: 'user',
              content: [
                { type: 'video', data: await sampleVideo(clip, { maxFrames }) },
                { type: 'text', text: 'Describe this video' },
              ],
            },
          ],
          max_tokens: 4,
        });
        expect(res.choices[0].message.content).toBeTypeOf('string');
        return res.usage!.prompt_tokens;
      };
      const nativeOne = await nativePromptTokens(1);
      const nativeThree = await nativePromptTokens(3);
      expect(nativeThree).toBeGreaterThan(nativeOne * 2);

      // a video file is not a valid input, the browser must decode it first
      await expect(
        wllama.createChatCompletion({
          messages: [
            {
              role: 'user',
              content: [{ type: 'video', data: await clip.arrayBuffer() }],
            },
          ],
          max_tokens: 4,
        })
      ).rejects.toThrow();

      await wllama.exit();
    },
    180000
  );

  test.sequential(
    'sends the frames to a model that merges video frames',
    async () => {
      const wllama = new Wllama({ default: '/src/wasm/wllama.wasm' });
      wllama.setCompat(null);
      await wllama.loadModelFromUrl(MERGE_MODEL, {
        n_ctx: 4096,
        n_threads: 4,
      });
      expect(wllama.supportInputModality('video')).toBe(true);

      const promptTokens = async (maxFrames: number) => {
        const res = await wllama.createChatCompletion({
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'video',
                  data: await sampleVideo(clip, { maxFrames, maxEdge: 128 }),
                },
                { type: 'text', text: 'Describe this video' },
              ],
            },
          ],
          max_tokens: 4,
        });
        return res.usage!.prompt_tokens;
      };

      // 2 frames are 1 group, 3 or 4 frames are 2 groups
      const two = await promptTokens(2);
      const three = await promptTokens(3);
      const four = await promptTokens(4);
      expect(four).toBeGreaterThan(two);
      expect(three).toBe(four);

      await wllama.exit();
    },
    180000
  );
});
