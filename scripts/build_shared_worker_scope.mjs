// Bundles src/workers-code/shared-worker-scope.ts into a single classic script, ready to run inside a SharedWorker.
// Usage: node ./scripts/build_shared_worker_scope.mjs <output-file>
//
// Heavy modules are replaced by stubs (see scripts/shared-worker-scope-stubs), the real code strings travel from the tab to the scope at runtime instead of being bundled twice.

import * as esbuild from 'esbuild';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const STUBS = path.join(ROOT, 'scripts', 'shared-worker-scope-stubs');

const outfile = process.argv[2];
if (!outfile) {
  console.error('usage: node build_shared_worker_scope.mjs <output-file>');
  process.exit(1);
}

const stubPlugin = {
  name: 'shared-worker-scope-stubs',
  setup(build) {
    build.onResolve({ filter: /workers-code\/generated$/ }, () => ({
      path: path.join(STUBS, 'generated.ts'),
    }));
    build.onResolve({ filter: /^\.\.?\/debug$/ }, () => ({
      path: path.join(STUBS, 'debug.ts'),
    }));
  },
};

await esbuild.build({
  entryPoints: [path.join(ROOT, 'src', 'workers-code', 'shared-worker-scope.ts')],
  bundle: true,
  minify: true,
  format: 'iife',
  target: 'es2022',
  outfile,
  plugins: [stubPlugin],
  logLevel: 'warning',
});

console.log('shared-worker-scope bundle written to ' + outfile);
