// Build-time stub for debug.ts, used only when bundling shared-worker-scope.ts
// The real debug.ts embeds the wasm source map (100KB+); inside the shared worker scope we keep raw stack traces instead.

export const Debug = {
  decodeStackTrace: async (stack: string, _isCompatBuild: boolean) => stack,
  decodeFuncIds: async (_funcIds: number[], _isCompatBuild: boolean) => [],
};
