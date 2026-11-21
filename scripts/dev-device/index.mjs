#!/usr/bin/env node
// Dev server to test wllama on a phone or tablet, driven by an agent on this machine. See "Testing on a phone or tablet" in README-dev.md.
//
//   node scripts/dev-device/index.mjs [serve] [--port 8090] [--models DIR] [--root DIR] [--no-watch] [--adb]
//   node scripts/dev-device/index.mjs status | open <path> | reload [--hard] | eval <code> | run <file.js> | logs [--follow]
//
// The device opens /device: a page that stays connected (server-sent events) and keeps the screen awake,
// with the page under test in an iframe. The agent sends commands with this CLI (or POST /api/cmd), the device runs them in the iframe
// and sends back the result and the console output.

import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// --root: serve another checkout (a worktree), the device page and the certificates stay here
const ROOT = path.resolve(process.argv.includes('--root') ? process.argv[process.argv.indexOf('--root') + 1] : path.resolve(HERE, '../..'));
const CERT_DIR = path.join(HERE, '.certs');

const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const opt = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : def;
};
const PORT = Number(opt('--port', process.env.WLLAMA_DEV_PORT || 8090));
const positional = argv.filter((a, i) => !a.startsWith('--') && !(i > 0 && ['--port', '--models', '--timeout', '--device', '--root'].includes(argv[i - 1])));
const command = positional[0] || 'serve';

//////////////////////////////////////////////////////////////
// CLI (agent side): talks to the running server on localhost
//////////////////////////////////////////////////////////////

// node:http, not fetch: fetch gives up waiting for the response headers after 300s
const api = (method, url, body) =>
  new Promise((resolve) => {
    const data = body ? JSON.stringify(body) : undefined;
    const req = http.request({ host: '127.0.0.1', port: PORT, path: url, method, headers: data ? { 'content-type': 'application/json' } : {} }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const json = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        if (res.statusCode >= 400 || json.ok === false) {
          console.log(JSON.stringify(json, null, 2));
          process.exit(1);
        }
        resolve(json);
      });
    });
    req.on('error', (e) => {
      console.error(`cannot reach the dev server on port ${PORT} (${e.code || e.message}), start it with: npm run dev:device`);
      process.exit(2);
    });
    req.end(data);
  });

const sendCmd = async (cmd) => {
  const timeout = Number(opt('--timeout', 120)) * 1000;
  const device = opt('--device', undefined);
  const res = await api('POST', '/api/cmd', { ...cmd, timeout, device });
  console.log(JSON.stringify(res, null, 2));
};

if (command !== 'serve') {
  switch (command) {
    case 'status':
      console.log(JSON.stringify(await api('GET', '/api/status'), null, 2));
      break;
    case 'open':
      await sendCmd({ type: 'open', url: positional[1] || '/scripts/dev-device/blank.html' });
      break;
    case 'reload':
      await sendCmd({ type: 'reload', hard: flag('--hard') });
      break;
    case 'eval':
      await sendCmd({ type: 'eval', code: positional.slice(1).join(' ') });
      break;
    case 'run':
      await sendCmd({ type: 'eval', code: fs.readFileSync(positional[1], 'utf8') });
      break;
    case 'logs': {
      let since = Number(opt('--since', 0));
      do {
        const { logs, next } = await api('GET', `/api/logs?since=${since}`);
        for (const l of logs) console.log(`${new Date(l.t).toISOString().slice(11, 23)} [${l.device}] ${l.level}: ${l.text}`);
        since = next;
        if (flag('--follow')) await new Promise((r) => setTimeout(r, 1000));
      } while (flag('--follow'));
      break;
    }
    default:
      console.error(`unknown command: ${command}`);
      process.exit(2);
  }
  process.exit(0);
}

//////////////////////////////////////////////////////////////
// server
//////////////////////////////////////////////////////////////

const MODELS_DIR = opt('--models', undefined);
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.cjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/plain; charset=utf-8',
};

const lanIps = () =>
  Object.values(os.networkInterfaces())
    .flat()
    .filter((i) => i && i.family === 'IPv4' && !i.internal)
    .map((i) => i.address);

// state
const devices = new Map(); // id -> { id, res (SSE), info, connectedAt, lastSeen }
const pending = new Map(); // cmd id -> { resolve, timer }
const logs = []; // { n, t, device, level, text }
let logSeq = 0;
let cmdSeq = 0;
const MAX_LOGS = 5000;

const addLog = (device, level, text) => {
  logs.push({ n: ++logSeq, t: Date.now(), device, level, text });
  if (logs.length > MAX_LOGS) logs.splice(0, logs.length - MAX_LOGS);
  const short = text.length > 500 ? text.slice(0, 500) + '...' : text;
  console.log(`[${device}] ${level}: ${short}`);
};

const sendEvent = (dev, msg) => dev.res.write(`data: ${JSON.stringify(msg)}\n\n`);

const pickDevice = (id) => {
  if (id) return devices.get(id);
  // a visible page first (the browser throttles hidden tabs), then the device that connected last
  return [...devices.values()].sort((a, b) => (b.visible ? 1 : 0) - (a.visible ? 1 : 0) || b.connectedAt - a.connectedAt)[0];
};

const json = (res, code, body) => {
  res.writeHead(code, { 'content-type': 'application/json', 'cache-control': 'no-store' });
  res.end(JSON.stringify(body));
};

const readBody = (req) =>
  new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {});
      } catch (e) {
        reject(e);
      }
    });
    req.on('error', reject);
  });

const isLocal = (req) => ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(req.socket.remoteAddress);

// same headers as the vitest config, so multi-thread (SharedArrayBuffer) works
const ISOLATION_HEADERS = {
  'cross-origin-opener-policy': 'same-origin',
  'cross-origin-embedder-policy': 'require-corp',
  'cross-origin-resource-policy': 'same-origin',
  'cache-control': 'no-store',
};

const serveFile = (req, res, file) => {
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, ISOLATION_HEADERS);
      return res.end('not found');
    }
    const headers = {
      ...ISOLATION_HEADERS,
      'content-type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
      'accept-ranges': 'bytes',
    };
    // Range requests, to read big model files in parts
    const m = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
    if (m) {
      const start = m[1] ? Number(m[1]) : st.size - Number(m[2]);
      const end = m[1] && m[2] ? Math.min(Number(m[2]), st.size - 1) : st.size - 1;
      if (start > end || start >= st.size) {
        res.writeHead(416, { ...headers, 'content-range': `bytes */${st.size}` });
        return res.end();
      }
      res.writeHead(206, { ...headers, 'content-range': `bytes ${start}-${end}/${st.size}`, 'content-length': end - start + 1 });
      if (req.method === 'HEAD') return res.end();
      return fs.createReadStream(file, { start, end }).pipe(res);
    }
    res.writeHead(200, { ...headers, 'content-length': st.size });
    if (req.method === 'HEAD') return res.end();
    fs.createReadStream(file).pipe(res);
  });
};

// resolve a URL path under a base directory; no dot files (.git, .env, .certs) and no way out of the base
const safeJoin = (base, urlPath) => {
  const rel = decodeURIComponent(urlPath).replace(/^\/+/, '');
  if (rel.split('/').some((seg) => seg.startsWith('.'))) return null;
  const file = path.resolve(base, rel);
  return file === base || file.startsWith(base + path.sep) ? file : null;
};

const INDEX_HTML = () => `<!doctype html><meta name="viewport" content="width=device-width">
<title>wllama dev</title><body style="font-family:sans-serif;padding:16px">
<p><a href="/device"><b>Open the device page</b></a> (the agent drives it)</p>
<p>Examples: <a href="/examples/basic/">basic</a>, <a href="/examples/main/dist/">main</a>, <a href="/examples/test-backend-ops/">test-backend-ops</a></p>
${MODELS_DIR ? '<p>Models from this machine are in <a href="/models/">/models/</a></p>' : ''}`;

const handler = async (req, res) => {
  const url = new URL(req.url, 'http://x');
  const p = url.pathname;
  try {
    // device side
    if (p === '/device') return serveFile(req, res, path.join(HERE, 'device.html'));
    if (p === '/favicon.ico') {
      res.writeHead(204);
      return res.end();
    }
    if (p === '/api/events') {
      const id = url.searchParams.get('id') || 'device';
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store', connection: 'keep-alive' });
      res.write(': connected\n\n');
      const old = devices.get(id);
      const dev = { id, res, info: old?.info ?? null, connectedAt: Date.now(), lastSeen: Date.now() };
      devices.set(id, dev);
      addLog(id, 'server', `connected from ${req.socket.remoteAddress}`);
      const ping = setInterval(() => res.write(': ping\n\n'), 15000);
      req.on('close', () => {
        clearInterval(ping);
        if (devices.get(id) === dev) {
          devices.delete(id);
          addLog(id, 'server', 'disconnected');
        }
      });
      return;
    }
    if (p === '/api/hello' && req.method === 'POST') {
      const { id, info } = await readBody(req);
      const dev = devices.get(id);
      if (dev) dev.info = info;
      addLog(id, 'server', `info: ${JSON.stringify(info)}`);
      return json(res, 200, { ok: true });
    }
    if (p === '/api/state' && req.method === 'POST') {
      const { id, visible } = await readBody(req);
      const dev = devices.get(id);
      if (dev) dev.visible = visible;
      return json(res, 200, { ok: true });
    }
    if (p === '/api/log' && req.method === 'POST') {
      const { id, entries } = await readBody(req);
      for (const e of entries || []) addLog(id, e.level, e.text);
      const dev = devices.get(id);
      if (dev) dev.lastSeen = Date.now();
      return json(res, 200, { ok: true });
    }
    if (p === '/api/result' && req.method === 'POST') {
      const body = await readBody(req);
      const w = pending.get(body.cmd);
      if (w) {
        clearTimeout(w.timer);
        pending.delete(body.cmd);
        w.resolve(body);
      }
      return json(res, 200, { ok: true });
    }

    // agent side, only from this machine
    if (p === '/api/status' || p === '/api/cmd' || p === '/api/logs') {
      if (!isLocal(req)) return json(res, 403, { ok: false, error: 'only from localhost' });
      if (p === '/api/status') {
        return json(res, 200, {
          ok: true,
          devices: [...devices.values()].map((d) => ({ id: d.id, visible: !!d.visible, info: d.info, connectedAt: new Date(d.connectedAt).toISOString() })),
        });
      }
      if (p === '/api/logs') {
        const since = Number(url.searchParams.get('since') || 0);
        const out = logs.filter((l) => l.n > since);
        return json(res, 200, { ok: true, logs: out, next: out.length ? out[out.length - 1].n : since });
      }
      // /api/cmd: forward to a device and wait for its result
      const cmd = await readBody(req);
      const dev = pickDevice(cmd.device);
      if (!dev) return json(res, 404, { ok: false, error: 'no device connected, open /device on the phone' });
      const id = ++cmdSeq;
      const timeout = cmd.timeout || 120000;
      const result = await new Promise((resolve) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          resolve({ ok: false, error: `timeout after ${timeout / 1000}s` });
        }, timeout);
        pending.set(id, { resolve, timer });
        sendEvent(dev, { ...cmd, id, device: undefined, timeout: undefined });
      });
      return json(res, 200, { device: dev.id, ...result });
    }

    // static files
    if (p === '/') {
      res.writeHead(200, { ...ISOLATION_HEADERS, 'content-type': 'text/html; charset=utf-8' });
      return res.end(INDEX_HTML());
    }
    if (MODELS_DIR && p.startsWith('/models/')) {
      const base = path.resolve(MODELS_DIR);
      if (p === '/models/') {
        const files = fs.readdirSync(base).filter((f) => !f.startsWith('.'));
        res.writeHead(200, { ...ISOLATION_HEADERS, 'content-type': 'text/html; charset=utf-8' });
        return res.end(files.map((f) => `<a href="/models/${encodeURIComponent(f)}">${f}</a>`).join('<br>'));
      }
      const file = safeJoin(base, p.slice('/models/'.length));
      if (!file) return json(res, 403, { ok: false });
      return serveFile(req, res, fs.existsSync(file) ? fs.realpathSync(file) : file);
    }
    let file = safeJoin(ROOT, p);
    if (!file) return json(res, 403, { ok: false });
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
    return serveFile(req, res, file);
  } catch (e) {
    json(res, 500, { ok: false, error: String(e) });
  }
};

// self-signed certificate for the LAN addresses, so the page is a secure context without adb (the browser warns once)
const getCert = () => {
  const ips = ['127.0.0.1', ...lanIps()];
  const keyFile = path.join(CERT_DIR, 'key.pem');
  const certFile = path.join(CERT_DIR, 'cert.pem');
  const ipsFile = path.join(CERT_DIR, 'ips.txt');
  const want = ips.join(',');
  if (!fs.existsSync(certFile) || !fs.existsSync(ipsFile) || fs.readFileSync(ipsFile, 'utf8') !== want) {
    fs.mkdirSync(CERT_DIR, { recursive: true });
    const san = ['DNS:localhost', ...ips.map((ip) => `IP:${ip}`)].join(',');
    execFileSync('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825', '-subj', '/CN=wllama-dev', '-addext', `subjectAltName=${san}`, '-keyout', keyFile, '-out', certFile], { stdio: 'ignore' });
    fs.writeFileSync(ipsFile, want);
  }
  return { key: fs.readFileSync(keyFile), cert: fs.readFileSync(certFile) };
};

// reload the page under test on the devices when a served file changes
const watch = () => {
  const IGNORE = /(^|\/)(node_modules|llama\.cpp|build[^/]*|\.git|\.certs|tmp|cache|docs|test-results)(\/|$)|\.DS_Store$/;
  let timer = null;
  const changed = new Set();
  try {
    fs.watch(ROOT, { recursive: true }, (_ev, f) => {
      if (!f || IGNORE.test(f)) return;
      changed.add(f);
      clearTimeout(timer);
      timer = setTimeout(() => {
        const files = [...changed];
        changed.clear();
        addLog('server', 'server', `changed: ${files.slice(0, 5).join(', ')}${files.length > 5 ? ', ...' : ''}, reloading the devices`);
        // a change of the device page itself reloads it, other changes reload the page under test
        const hard = files.some((f) => f.endsWith('dev-device/device.html'));
        for (const dev of devices.values()) sendEvent(dev, { type: 'reload', hard, id: 0 });
      }, 300);
    });
  } catch (e) {
    console.warn(`file watching is not available: ${e.message}`);
  }
};

http.createServer(handler).listen(PORT, '0.0.0.0');
let httpsPort = null;
try {
  https.createServer(getCert(), handler).listen(PORT + 1, '0.0.0.0');
  httpsPort = PORT + 1;
} catch (e) {
  console.warn(`no HTTPS (${e.message}), only adb reverse will give a secure context`);
}
if (!flag('--no-watch')) watch();

// --adb: for a device on USB (or wireless debugging): forward the port, keep the screen on while it is plugged in
// (the browser wake lock is not always enough, for example in power saving), and open the device page in Chrome
if (flag('--adb')) {
  const adb = (...args) => {
    try {
      return execFileSync('adb', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
    } catch (e) {
      console.warn(`adb ${args.join(' ')}: ${(e.stderr || e.message).toString().trim()}`);
      return null;
    }
  };
  const model = adb('shell', 'getprop', 'ro.product.model');
  if (model !== null) {
    // applied again every 15s: a replugged cable, or another server stopping, removes them
    const apply = () => {
      if (!(adb('reverse', '--list') || '').includes(`tcp:${PORT}`)) adb('reverse', `tcp:${PORT}`, `tcp:${PORT}`);
      adb('shell', 'svc', 'power', 'stayon', 'usb');
    };
    apply();
    setInterval(apply, 15000);
    adb('shell', 'input', 'keyevent', 'KEYCODE_WAKEUP');
    // the application id makes Chrome reuse the same tab
    adb('shell', 'am', 'start', '-a', 'android.intent.action.VIEW', '-d', `http://localhost:${PORT}/device`, '-e', 'com.android.browser.application_id', 'wllama-dev', 'com.android.chrome');
    console.log(`adb: ${model}: port forwarded, screen kept on while plugged in, device page opened (it connects once the device is unlocked)`);
    const restore = () => {
      adb('shell', 'svc', 'power', 'stayon', 'false');
      adb('reverse', '--remove', `tcp:${PORT}`);
      process.exit(0);
    };
    process.on('SIGINT', restore);
    process.on('SIGTERM', restore);
  }
}

console.log(`wllama device server
  with adb (USB or wireless debugging), recommended:
    adb reverse tcp:${PORT} tcp:${PORT}   then open on the device: http://localhost:${PORT}/device
  without adb (same Wi-Fi, accept the certificate warning once):
${httpsPort ? lanIps().map((ip) => `    https://${ip}:${httpsPort}/device`).join('\n') : '    (HTTPS is not available)'}
  agent: node scripts/dev-device/index.mjs status | open <path> | reload | eval <code> | run <file.js> | logs
${MODELS_DIR ? `  models: ${path.resolve(MODELS_DIR)} is served at /models/\n` : ''}`);
