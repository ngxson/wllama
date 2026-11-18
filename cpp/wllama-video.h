#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "mtmd-helper.h"

// Video decoder for mtmd. The browser decodes the video (see sampleVideo() in src/video.ts), this code only reads the frames
// Layout: header, then n_frames * width * height * 3 bytes of RGB24. All values are little-endian

#define WLLAMA_VIDEO_MAGIC "WLVF"
#define WLLAMA_VIDEO_VERSION 1

struct wllama_video_header
{
  char magic[4];
  uint32_t version;
  uint32_t width;
  uint32_t height;
  uint32_t n_frames;
  float fps;
};

struct wllama_video_state
{
  wllama_video_header header;
  std::vector<unsigned char> frames;
  uint32_t i_frame = 0;
};

// bilinear
static void wllama_video_resize(const unsigned char *src, size_t sw, size_t sh, unsigned char *dst, size_t dw, size_t dh)
{
  const float scale_x = (float)sw / dw;
  const float scale_y = (float)sh / dh;
  for (size_t y = 0; y < dh; y++)
  {
    const float fy = std::max(0.0f, (y + 0.5f) * scale_y - 0.5f);
    const size_t y0 = std::min((size_t)fy, sh - 1);
    const size_t y1 = std::min(y0 + 1, sh - 1);
    const float wy = fy - y0;
    for (size_t x = 0; x < dw; x++)
    {
      const float fx = std::max(0.0f, (x + 0.5f) * scale_x - 0.5f);
      const size_t x0 = std::min((size_t)fx, sw - 1);
      const size_t x1 = std::min(x0 + 1, sw - 1);
      const float wx = fx - x0;
      for (size_t c = 0; c < 3; c++)
      {
        const float top = src[(y0 * sw + x0) * 3 + c] * (1.0f - wx) + src[(y0 * sw + x1) * 3 + c] * wx;
        const float bottom = src[(y1 * sw + x0) * 3 + c] * (1.0f - wx) + src[(y1 * sw + x1) * 3 + c] * wx;
        dst[(y * dw + x) * 3 + c] = (unsigned char)(top * (1.0f - wy) + bottom * wy + 0.5f);
      }
    }
  }
}

// returns nullptr for any other file, so images and audio keep their own decoders
static void *wllama_video_open(void *, const char *, const unsigned char *buf, size_t len, mtmd_helper_video_stream_info *info)
{
  wllama_video_header h;
  if (buf == nullptr || len < sizeof(h))
    return nullptr;
  memcpy(&h, buf, sizeof(h));
  if (memcmp(h.magic, WLLAMA_VIDEO_MAGIC, sizeof(h.magic)) != 0 || h.version != WLLAMA_VIDEO_VERSION)
    return nullptr;

  const uint64_t n_bytes = len - sizeof(h);
  const uint64_t n_pixels = (uint64_t)h.width * h.height;
  const float duration = h.n_frames / h.fps;
  if (n_pixels == 0 || h.n_frames == 0 || !(h.fps > 0.0f) || !std::isfinite(h.fps) || !std::isfinite(duration))
    return nullptr;
  if (n_pixels > n_bytes / 3 || n_bytes % (n_pixels * 3) != 0 || n_bytes / (n_pixels * 3) != h.n_frames)
    return nullptr;

  auto st = std::make_unique<wllama_video_state>();
  st->header = h;
  st->frames.assign(buf + sizeof(h), buf + len); // buf is only valid during this call

  info->width = h.width;
  info->height = h.height;
  info->fps = h.fps;
  info->n_frames = h.n_frames;
  info->duration = duration;
  return st.release();
}

static bool wllama_video_read_frame(void *state, unsigned char *rgb, uint32_t width, uint32_t height)
{
  auto *st = static_cast<wllama_video_state *>(state);
  const auto &h = st->header;
  if (st->i_frame >= h.n_frames)
    return false;
  const size_t frame_size = (size_t)h.width * h.height * 3;
  const unsigned char *src = st->frames.data() + st->i_frame * frame_size;
  st->i_frame++;
  if (rgb == nullptr)
    return true; // skip this frame
  if (width == h.width && height == h.height)
    memcpy(rgb, src, frame_size);
  else
    wllama_video_resize(src, h.width, h.height, rgb, width, height);
  return true;
}

static void wllama_video_close(void *state)
{
  delete static_cast<wllama_video_state *>(state);
}

static const mtmd_helper_video_decoder wllama_video_decoder = {
    /* open       */ wllama_video_open,
    /* start      */ nullptr,
    /* read_frame */ wllama_video_read_frame,
    /* read_audio */ nullptr,
    /* close      */ wllama_video_close,
    /* user_data  */ nullptr,
};
