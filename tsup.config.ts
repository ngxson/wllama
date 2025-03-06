import type { Options } from 'tsup';

const baseConfig: Options = {
  entry: ['./index.ts'],
  format: ['cjs', 'esm'],
  outDir: 'esm',
  clean: true,
};

// const nodeConfig: Options = {
//   ...baseConfig,
//   platform: "node",
// };

const browserConfig: Options = {
  ...baseConfig,
  platform: 'browser',
  target: 'es2015',
  splitting: false,
  outDir: 'esm',
};

export default [browserConfig];
