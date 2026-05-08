# Introducing Wllama V3.0

## What's new

V3.0 is a major architectural overhaul that replaces the custom wllama core with `server-context`, the inference component from `llama-server`. Key highlights:

- Full OAI-compatible API: `createChatCompletion`, `createCompletion`, `createEmbedding`
- Multimodal support (vision/audio inputs)
- Native tool calling support
- Jinja-based chat template parsing (same as llama-server)

New demos:
- Multimodal (vision) completion: https://github.ngxson.com/wllama/examples/multimodal/ ([source code](../examples/multimodal/index.html))
- Tool calling: https://github.ngxson.com/wllama/examples/tools/ ([source code](../examples/tools/index.html))

## New architecture

Previously, wllama implemented its own low-level bindings to llama.cpp. V3.0 instead reuses `server-context.cpp` from `llama-server`, which brings two major benefits:

- Better compatibility: new llama.cpp features (tool calling, reasoning, multimodal) work automatically
- Less maintenance: wllama no longer needs to re-implement chat template parsing, sampling logic, etc.

The worker architecture is unchanged — the wasm thread runs the server-context main loop, the browser thread handles inference requests, and they communicate via the existing `glue` message protocol.

## OAI-compatible API

All completion methods now follow the OpenAI API shape closely. This makes it easy to swap wllama in wherever you already use the OpenAI SDK.

### `createChatCompletion`

```typescript
// Non-streaming
const response = await wllama.createChatCompletion({
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 256,
  temperature: 0.7,
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await wllama.createChatCompletion({
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 256,
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0].delta.content ?? '');
}
```

### `createCompletion`

```typescript
// Raw (non-chat) completion
const response = await wllama.createCompletion({
  prompt: 'The capital of France is',
  max_tokens: 32,
});
console.log(response.choices[0].text);
```

### `createEmbedding`

```typescript
// Requires model loaded with { embeddings: true }
const response = await wllama.createEmbedding({
  input: 'The quick brown fox',
});
console.log(response.data[0].embedding); // float[]
```

## Tool calling

Tool calling works out of the box for any model that supports it (e.g. Qwen, Llama with tool-call template).

```typescript
const tools = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather for a given city.',
      parameters: {
        type: 'object',
        properties: {
          city: { type: 'string', description: 'City name' },
        },
        required: ['city'],
      },
    },
  },
];

const messages = [{ role: 'user', content: 'What is the weather in Tokyo?' }];

// First turn: model decides to call a tool
const response = await wllama.createChatCompletion({
  messages,
  tools,
  tool_choice: 'auto',
  max_tokens: 256,
});

const choice = response.choices[0];
if (choice.finish_reason === 'tool_calls') {
  const toolCall = choice.message.tool_calls[0];
  const args = JSON.parse(toolCall.function.arguments);
  const result = { condition: 'rain', temperature_celsius: 21 };

  // Second turn: feed tool result back
  messages.push(choice.message);
  messages.push({
    role: 'tool',
    tool_call_id: toolCall.id,
    content: JSON.stringify(result),
  });

  const final = await wllama.createChatCompletion({ messages, max_tokens: 256 });
  console.log(final.choices[0].message.content);
}
```

## Multimodal support

Models with a vision projector (mmproj) can now process image and audio inputs.

```typescript
// Load the model + mmproj from Hugging Face
await wllama.loadModelFromHF({
  repo: 'user/model-GGUF',
  quant: 'Q4_K_M',
  mmprojQuant: 'Q8_0',
});

// Or load from explicit URLs
await wllama.loadModelFromUrl({
  url: 'https://example.com/model.gguf',
  mmprojUrl: 'https://example.com/mmproj.gguf',
});

// Pass an image as ArrayBuffer alongside text
const imageData = await fetch('./photo.jpg').then(r => r.arrayBuffer());

const response = await wllama.createChatCompletion({
  messages: [
    {
      role: 'user',
      content: [
        { type: 'image', data: imageData },
        { type: 'text', text: 'Describe this image.' },
      ],
    },
  ],
  max_tokens: 512,
});
```

## Migration from v2.0

### Removed low-level APIs

The following APIs are **no longer available** in v3.0. They were tied to the old custom core and cannot easily be re-implemented on top of llama-server.

| Removed | Reason |
|---|---|
| `tokenize` / `detokenize` | Low-level tokenizer API removed |
| `decode` / `encode` | Replaced by OAI completion API |
| `samplingInit` / `samplingAccept` / `samplingSample` | Sampling is now handled internally per-request |
| Sequence shift/remove operations | Not exposed by llama-server context |

> [!IMPORTANT]
> If you rely on tokenizer APIs, please leave a comment on [PR #213](https://github.com/ngxson/wllama/pull/213) — it can be added back in the future.

### Sampling params moved to per-request

In v2.0, some sampling params were passed at model load time. From v3.0, all sampling params must be provided per request via `createChatCompletion` / `createCompletion`.

Previously in v2.x:

```js
await wllama.loadModelFromUrl('https://example.com/model.gguf', {
  temperature: 0.8,
  top_k: 40,
});
```

From v3.0:

```js
await wllama.loadModelFromUrl('https://example.com/model.gguf');

const response = await wllama.createChatCompletion({
  messages: [{ role: 'user', content: 'Hello!' }],
  temperature: 0.8,
  top_k: 40,
});
```

### Auto context length removed

The `n_ctx_auto` option is no longer supported. Set `n_ctx` explicitly at load time.

```js
await wllama.loadModelFromUrl('https://example.com/model.gguf', {
  n_ctx: 4096,
});
```

### Multimodal loading: `mmprojUrl` replaces separate file selection

Previously you had to pass mmproj as a local file alongside the main model. From v3.0, pass it directly via `mmprojUrl` in `loadModelFromUrl`:

```js
await wllama.loadModelFromUrl({
  url: 'https://example.com/model.gguf',
  mmprojUrl: 'https://example.com/mmproj.gguf',
});
```

Local file loading still works — just pass both GGUF blobs to `loadModel`:

```js
await wllama.loadModel([modelBlob, mmprojBlob]);
```

### Internal changes

- The `server-context` main loop now runs on the wasm worker thread
- Chat templates are parsed with Jinja (same as llama-server) — set `jinja: true` at load time to enable, or override with `chat_template`
- `WllamaError` gains a new `'kv_cache_full'` error type for when the context runs out of space
