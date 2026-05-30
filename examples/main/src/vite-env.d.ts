/// <reference types="vite/client" />

declare module 'virtual:wllama-compat' {
  import type { WllamaCompat } from '@wllama/wllama';
  const config: WllamaCompat | 'default';
  export default config;
}
