import { defineConfig } from 'vitest/config';

const SAFARI = process.env.BROWSER === 'safari';
const WEBGPU = process.env.WEBGPU === '1';
const AUTO = process.env.AUTO === '1';
const HEADED = process.env.HEADED === '1';
// run src/testing/wllama.test.ts on the compat build (asyncify), see vitest.setup.ts
const COMPAT = process.env.COMPAT === '1';
// run src/testing/big-model.test.ts only (6GB model made on the fly, needs WebGPU and RAM)
const BIG_MODEL = process.env.BIG_MODEL === '1';

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
    __HF_TOKEN__: JSON.stringify(process.env.HF_TOKEN ?? ''),
    __COMPAT__: JSON.stringify(COMPAT),
  },
  test: {
    setupFiles: ['./vitest.setup.ts'],
    ...(AUTO ? { watch: false } : {}),
    exclude: [
      '**/node_modules/**',
      '**/esm/**',
      '**/docs/**',
      '**/examples/**',
      ...(!WEBGPU ? ['**/src/testing/*.wgpu.test.*'] : []),
      ...(!BIG_MODEL ? ['**/src/testing/big-model.test.*'] : []),
    ],
    include: BIG_MODEL
      ? ['**/src/testing/big-model.test.*']
      : WEBGPU
        ? ['**/src/testing/*.wgpu.test.*']
        : COMPAT
          ? [
              '**/src/testing/wllama.test.*',
              '**/src/testing/rpc.test.*',
              '**/src/testing/opfs-read.test.*',
            ]
          : ['**/src/testing/*.test.*'],
    browser: {
      enabled: true,
      name: process.env.BROWSER ?? 'chromium',
      provider: SAFARI ? 'webdriverio' : 'playwright',
      // safaridriver has no headless mode
      headless: !SAFARI && !HEADED,
      // https://playwright.dev
      providerOptions:
        WEBGPU || BIG_MODEL
          ? {
              launch: {
                // full chromium in new headless mode, the default headless shell has no WebGPU
                channel: 'chromium',
                args: chromeArgsWebGPU.map((a) => `--${a}`),
              },
            }
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
