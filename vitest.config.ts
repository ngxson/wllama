import { defineConfig } from 'vitest/config';

const SAFARI = process.env.BROWSER === 'safari';
const WEBGPU = process.env.WEBGPU === '1';
const AUTO = process.env.AUTO === '1';
const LORA_INTEGRATION = process.env.LORA_INTEGRATION === '1';
const GAME_BASE_INTEGRATION = process.env.GAME_BASE_INTEGRATION === '1';
const QWEN35_INTEGRATION = process.env.QWEN35_INTEGRATION === '1';

const chromeArgsCI = ['disable-gpu', 'no-sandbox', 'disable-setuid-sandbox'];
const chromeArgsWebGPU = [
  'no-sandbox',
  'disable-setuid-sandbox',
  'enable-unsafe-webgpu',
  'enable-features=WebGPU',
];

export default defineConfig({
  define: {
    __GITHUB_CI__: JSON.stringify(!!process.env.GITHUB_ACTIONS),
    __LORA_INTEGRATION__: JSON.stringify(LORA_INTEGRATION),
    __GAME_BASE_INTEGRATION__: JSON.stringify(GAME_BASE_INTEGRATION),
    __QWEN35_INTEGRATION__: JSON.stringify(QWEN35_INTEGRATION),
  },
  test: {
    ...(AUTO ? { watch: false } : {}),
    exclude: [
      '**/node_modules/**',
      '**/esm/**',
      '**/docs/**',
      '**/examples/**',
      ...(!WEBGPU ? ['**/src/*.wgpu.test.*'] : []),
    ],
    include: WEBGPU ? ['**/src/*.wgpu.test.*'] : ['**/src/**/*.test.*'],
    browser: {
      enabled: true,
      name: process.env.BROWSER ?? 'chromium',
      provider: SAFARI ? 'webdriverio' : 'playwright',
      // https://playwright.dev
      providerOptions: WEBGPU
        ? { launch: { args: chromeArgsWebGPU.map((a) => `--${a}`) } }
        : process.env.GITHUB_ACTIONS
          ? {
              capabilities: {
                'goog:chromeOptions': {
                  args: chromeArgsCI,
                },
              },
            }
          : SAFARI
            ? {
                capabilities: {
                  alwaysMatch: { browserName: 'safari' },
                  firstMatch: [{}],
                  browserName: 'safari',
                },
              }
            : {},
    },
  },
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
});
