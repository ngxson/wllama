import type { ChatCompletionMessageContent } from './types/oai-compat';
import { WllamaError } from './wllama';

export interface SampleVideoOptions {
  /** Target sampling rate, in frames per second. With sampleVideo(), Qwen3-VL and LFM3 keep at most 2 frames per second. Default: 2 */
  fps?: number;
  /** Maximum number of frames. Each frame uses context tokens, so set this according to n_ctx. Default: 32 */
  maxFrames?: number;
  /** Maximum size of the longest edge, in pixels. Frames are never upscaled. Default: 512 */
  maxEdge?: number;
}

export interface SampledVideoFrame {
  /** Position of the frame in the video, in seconds */
  time: number;
  /** JPEG file content */
  data: ArrayBuffer;
}

const DEFAULT_FPS = 2;
const DEFAULT_MAX_FRAMES = 32;
const DEFAULT_MAX_EDGE = 512;
const MIN_FRAMES = 4; // same as the mtmd video config
const JPEG_QUALITY = 0.9;
const EVENT_TIMEOUT_MS = 30000;

// see cpp/wllama-video.h
const RAW_MAGIC = [0x57, 0x4c, 0x56, 0x46]; // "WLVF"
const RAW_VERSION = 1;
const RAW_HEADER_SIZE = 24;

/**
 * Get the evenly spaced positions to sample, in seconds. Position i is at i * duration / n, so the frame rate of the result is n / duration.
 */
export const getVideoSampleTimes = (
  duration: number,
  fps: number,
  maxFrames: number
): number[] => {
  const n = Math.min(
    Math.max(Math.floor(duration * fps), MIN_FRAMES),
    Math.floor(maxFrames)
  );
  return Array.from({ length: n }, (_, i) => (i * duration) / n);
};

const waitForEvent = (video: HTMLVideoElement, event: string): Promise<void> =>
  new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener(event, onEvent);
      video.removeEventListener('error', onError);
    };
    const onEvent = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      const reason = video.error?.message || 'unknown error';
      reject(
        new WllamaError(`Cannot read video: ${reason}`, 'inference_error')
      );
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(
        new WllamaError(`Timeout on video event: ${event}`, 'inference_error')
      );
    }, EVENT_TIMEOUT_MS);
    video.addEventListener(event, onEvent);
    video.addEventListener('error', onError);
  });

const seekTo = (video: HTMLVideoElement, time: number): Promise<void> => {
  const seeked = waitForEvent(video, 'seeked');
  video.currentTime = time;
  return seeked;
};

const encodeJpeg = (canvas: HTMLCanvasElement): Promise<Blob> =>
  new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) =>
        blob
          ? resolve(blob)
          : reject(
              new WllamaError('Cannot encode video frame', 'inference_error')
            ),
      'image/jpeg',
      JPEG_QUALITY
    );
  });

interface DrawnVideoFrame {
  /** The frame is drawn on the canvas of this context */
  ctx: CanvasRenderingContext2D;
  time: number;
  index: number;
  /** Number of frames in the result */
  count: number;
  /** Frame rate of the result */
  fps: number;
}

const readVideoFrames = async (
  source: Blob | string,
  options: SampleVideoOptions,
  onFrame: (frame: DrawnVideoFrame) => void | Promise<void>
): Promise<void> => {
  const fps = options.fps ?? DEFAULT_FPS;
  const maxFrames = options.maxFrames ?? DEFAULT_MAX_FRAMES;
  const maxEdge = options.maxEdge ?? DEFAULT_MAX_EDGE;
  if (!(fps > 0) || !(maxFrames >= 1) || !(maxEdge >= 1)) {
    throw new WllamaError(
      'fps, maxFrames and maxEdge must be positive',
      'inference_error'
    );
  }
  if (typeof document === 'undefined') {
    throw new WllamaError(
      'Video sampling needs the DOM, it cannot run in a web worker',
      'inference_error'
    );
  }

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true; // iOS does not decode inline video without this
  video.preload = 'auto';
  const isUrl = typeof source === 'string';
  if (isUrl) {
    video.crossOrigin = 'anonymous'; // or else the canvas becomes unreadable
  }
  const src = isUrl ? source : URL.createObjectURL(source);

  try {
    const loaded = waitForEvent(video, 'loadeddata');
    video.src = src;
    await loaded;

    if (!video.videoWidth || !video.videoHeight) {
      throw new WllamaError('File has no video track', 'inference_error');
    }
    // MediaRecorder files have no duration in the header, it is known only after a seek to the end
    if (!Number.isFinite(video.duration)) {
      await seekTo(video, Number.MAX_SAFE_INTEGER);
    }
    if (!Number.isFinite(video.duration) || video.duration <= 0) {
      throw new WllamaError('Cannot get video duration', 'inference_error');
    }

    const scale = Math.min(
      1,
      maxEdge / Math.max(video.videoWidth, video.videoHeight)
    );
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new WllamaError('Cannot create 2D canvas', 'inference_error');
    }

    const times = getVideoSampleTimes(video.duration, fps, maxFrames);
    for (let index = 0; index < times.length; index++) {
      await seekTo(video, times[index]);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      await onFrame({
        ctx,
        time: times[index],
        index,
        count: times.length,
        fps: times.length / video.duration,
      });
    }
  } finally {
    video.removeAttribute('src');
    video.load(); // release the decoder
    if (!isUrl) {
      URL.revokeObjectURL(src);
    }
  }
};

/**
 * Sample frames from a video with the video decoder of the browser. All formats that the browser can play are supported.
 *
 * This function needs the DOM, so it cannot run in a web worker. To read a URL from another origin, the server must allow CORS.
 *
 * @param source Video file, or URL of the video
 * @param options
 * @returns Frames in order of time
 */
export const sampleVideoFrames = async (
  source: Blob | string,
  options: SampleVideoOptions = {}
): Promise<SampledVideoFrame[]> => {
  const frames: SampledVideoFrame[] = [];
  await readVideoFrames(source, options, async ({ ctx, time }) => {
    const blob = await encodeJpeg(ctx.canvas);
    frames.push({ time, data: await blob.arrayBuffer() });
  });
  return frames;
};

/**
 * Sample a video for native video input: `{ type: 'video', data: await sampleVideo(file) }`. The model gets the frames in its own video layout, for example Qwen3-VL merges each 2 frames.
 *
 * The model must support video input, see `supportInputModality('video')`. The result holds raw RGB frames, so its size is width * height * 3 bytes per frame.
 *
 * This function has the same limits as sampleVideoFrames().
 *
 * @param source Video file, or URL of the video
 * @param options
 * @returns Data for a content part of type "video"
 */
export const sampleVideo = async (
  source: Blob | string,
  options: SampleVideoOptions = {}
): Promise<ArrayBuffer> => {
  let out = new Uint8Array(0);
  await readVideoFrames(source, options, ({ ctx, index, count, fps }) => {
    const { width, height } = ctx.canvas;
    const frameSize = width * height * 3;
    if (index === 0) {
      out = new Uint8Array(RAW_HEADER_SIZE + count * frameSize);
      out.set(RAW_MAGIC);
      const header = new DataView(out.buffer);
      header.setUint32(4, RAW_VERSION, true);
      header.setUint32(8, width, true);
      header.setUint32(12, height, true);
      header.setUint32(16, count, true);
      header.setFloat32(20, fps, true);
    }
    const rgba = ctx.getImageData(0, 0, width, height).data;
    let pos = RAW_HEADER_SIZE + index * frameSize;
    for (let i = 0; i < rgba.length; i += 4) {
      out[pos++] = rgba[i];
      out[pos++] = rgba[i + 1];
      out[pos++] = rgba[i + 2];
    }
  });
  return out.buffer;
};

/**
 * Sample a video into message content for createChatCompletion(). Each frame is an image part that follows its timestamp.
 *
 * The model must support image input. It reads the frames as separate images, so a model with its own video layout (for example Qwen3-VL) uses more tokens than with sampleVideo().
 *
 * @param source Video file, or URL of the video
 * @param options
 * @returns Content parts to add to the "content" of a message
 */
export const videoToMessageContent = async (
  source: Blob | string,
  options: SampleVideoOptions = {}
): Promise<ChatCompletionMessageContent[]> => {
  const frames = await sampleVideoFrames(source, options);
  return frames.flatMap(({ time, data }) => [
    { type: 'text' as const, text: `<${time.toFixed(1)} seconds>` },
    { type: 'image' as const, data },
  ]);
};
