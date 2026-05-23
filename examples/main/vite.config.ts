import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { existsSync } from 'fs';
import { resolve } from 'path';

const COMPAT_WASM = resolve(
  __dirname,
  'node_modules/@wllama/wllama-compat/wasm/wllama.wasm'
);
const COMPAT_JS = resolve(
  __dirname,
  'node_modules/@wllama/wllama-compat/wasm/wllama.js'
);
const compatAvailable = existsSync(COMPAT_WASM) && existsSync(COMPAT_JS);

// https://vitejs.dev/config/
export default defineConfig({
  base: './',
  plugins: [
    react(),
    {
      name: 'wllama-compat',
      resolveId(id) {
        if (id === 'virtual:wllama-compat') return '\0virtual:wllama-compat';
      },
      load(id) {
        if (id !== '\0virtual:wllama-compat') return;
        if (compatAvailable) {
          return `
import wasm from '${COMPAT_WASM}?url';
import worker from '${COMPAT_JS}?raw';
export default { wasm, worker: { code: worker } };
`;
        } else {
          console.warn(
            '[wllama-compat] compat WASM not found — falling back to CDN. Run "npm install" inside the compat package to build locally.'
          );
          return `export default 'default';`;
        }
      },
    },
    {
      name: 'isolation',
      configureServer(server) {
        server.middlewares.use((_req, res, next) => {
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
          res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
          next();
        });
      },
    },
  ],
});
