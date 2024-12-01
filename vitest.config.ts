import { defineConfig } from 'vitest/config';

const SAFARI = process.env.BROWSER === 'safari';

export default defineConfig({
  test: {
    exclude: [
      '**/node_modules/**',
      '**/esm/**',
      '**/docs/**',
      '**/examples/**',
    ],
    include: ['**/src/*.test.*'],
    browser: {
      enabled: true,
      name: process.env.BROWSER ?? 'chromium',
      provider: SAFARI ? 'webdriverio' : 'playwright',
      // https://playwright.dev
      providerOptions: process.env.GITHUB_ACTIONS
        ? {
            capabilities: {
              'goog:chromeOptions': {
                args: ['disable-gpu', 'no-sandbox', 'disable-setuid-sandbox'],
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
