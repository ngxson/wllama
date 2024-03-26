import express from 'express';
import path from 'path';
import mime from 'mime-types';
import { dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const app = express();

// Serve static files
app.use(express.static(path.join(__dirname, '..'), {
 setHeaders: (res) => {
    if (process.env.MULTITHREAD) {
      // add required security header to enable SharedArrayBuffer, needed to run multithread
      // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    }
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
 }
}));

if (!process.env.MULTITHREAD) {
  console.log('WARN: Running server without MULTITHREAD=1, this will effectively disable multithreading');
}

// Start the server
const PORT = 8080;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
