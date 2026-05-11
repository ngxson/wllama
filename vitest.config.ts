import { defineConfig } from 'vitest/config';

const SAFARI = process.env.BROWSER === 'safari';
const WEBGPU = process.env.WEBGPU === '1';

const chromeArgsCI = ['disable-gpu', 'no-sandbox', 'disable-setuid-sandbox'];
const chromeArgsWebGPU = [
  'no-sandbox',
  'disable-setuid-sandbox',
  'enable-unsafe-webgpu',
  'enable-features=WebGPU',
];

export default defineConfig({
  test: {
    exclude: [
      '**/node_modules/**',
      '**/esm/**',
      '**/docs/**',
      '**/examples/**',
      ...(!WEBGPU ? ['**/src/*.wgpu.test.*'] : []),
    ],
    include: WEBGPU ? ['**/src/*.wgpu.test.*'] : ['**/src/*.test.*'],
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
