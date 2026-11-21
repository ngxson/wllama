// transport for ggml-rpc between wasm instances, see cpp/wllama-rpc-transport.cpp
// this code runs in the main worker and in the pthread workers: on the client, ggml-rpc does its I/O in its own thread

// a channel links the client wasm to one server wasm, see cpp/wllama-rpc-transport.cpp
// int32 header: connection id, closed connection id, then head and tail of the client->server and server->client rings
const RPC_CONN = 0;
const RPC_CLOSED = 1;
const RPC_RING_IDX = [2, 3, 4, 5];
const RPC_HEADER_BYTES = 64;
const RPC_WAIT_MS = 100;
const rpcChannels = {}; // port -> channel

// main worker: Module.wasmMemory; pthread worker: the emscripten global wasmMemory
const rpcHeapU8 = () =>
  new Uint8Array(
    (typeof wasmMemory !== 'undefined' ? wasmMemory : Module.wasmMemory).buffer
  );

const rpcSetChannels = (channels) => {
  for (const { port, sab, role } of channels) {
    const cap = (sab.byteLength - RPC_HEADER_BYTES) / 2; // power of 2
    const c2s = {
      head: 2,
      tail: 3,
      data: new Uint8Array(sab, RPC_HEADER_BYTES, cap),
    };
    const s2c = {
      head: 4,
      tail: 5,
      data: new Uint8Array(sab, RPC_HEADER_BYTES + cap, cap),
    };
    rpcChannels[port] = {
      hdr: new Int32Array(sab, 0, RPC_HEADER_BYTES / 4),
      cap,
      tx: role === 'client' ? c2s : s2c,
      rx: role === 'client' ? s2c : c2s,
      served: 0,
    };
  }
};

const rpcIsOpen = (ch, conn) =>
  Atomics.load(ch.hdr, RPC_CONN) === conn &&
  Atomics.load(ch.hdr, RPC_CLOSED) !== conn;

const _wllama_rpc_has = (port) => (rpcChannels[port] ? 1 : 0);

// server: wait for the next connection from the client
const _wllama_rpc_accept = (port) => {
  const ch = rpcChannels[port];
  if (!ch) return 0;
  while (true) {
    const conn = Atomics.load(ch.hdr, RPC_CONN);
    if (conn > ch.served) {
      ch.served = conn;
      return conn;
    }
    Atomics.wait(ch.hdr, RPC_CONN, conn, RPC_WAIT_MS);
  }
};

// client: start a new connection with empty rings
const _wllama_rpc_connect = (port) => {
  const ch = rpcChannels[port];
  if (!ch) return 0;
  for (const i of RPC_RING_IDX) Atomics.store(ch.hdr, i, 0);
  const conn = Atomics.add(ch.hdr, RPC_CONN, 1) + 1;
  Atomics.notify(ch.hdr, RPC_CONN);
  return conn;
};

// head and tail count bytes and wrap at 2^31, (head - tail) | 0 is still the used size
const _wllama_rpc_send = (port, conn, ptr, size) => {
  const ch = rpcChannels[port];
  const { hdr, cap, tx } = ch;
  let done = 0;
  while (done < size) {
    if (!rpcIsOpen(ch, conn)) return 0;
    const head = Atomics.load(hdr, tx.head);
    const tail = Atomics.load(hdr, tx.tail);
    const free = cap - ((head - tail) | 0);
    if (free === 0) {
      Atomics.wait(hdr, tx.tail, tail, RPC_WAIT_MS);
      continue;
    }
    const pos = head & (cap - 1);
    const n = Math.min(free, size - done, cap - pos);
    tx.data.set(rpcHeapU8().subarray(ptr + done, ptr + done + n), pos);
    Atomics.store(hdr, tx.head, (head + n) | 0);
    Atomics.notify(hdr, tx.head);
    done += n;
  }
  return 1;
};

const _wllama_rpc_recv = (port, conn, ptr, size) => {
  const ch = rpcChannels[port];
  const { hdr, cap, rx } = ch;
  let done = 0;
  while (done < size) {
    const head = Atomics.load(hdr, rx.head);
    const tail = Atomics.load(hdr, rx.tail);
    const used = (head - tail) | 0;
    if (used === 0) {
      if (!rpcIsOpen(ch, conn)) return 0;
      Atomics.wait(hdr, rx.head, head, RPC_WAIT_MS);
      continue;
    }
    const pos = tail & (cap - 1);
    const n = Math.min(used, size - done, cap - pos);
    rpcHeapU8().set(rx.data.subarray(pos, pos + n), ptr + done);
    Atomics.store(hdr, rx.tail, (tail + n) | 0);
    Atomics.notify(hdr, rx.tail);
    done += n;
  }
  return 1;
};

const _wllama_rpc_close = (port, conn) => {
  const ch = rpcChannels[port];
  if (!ch) return;
  Atomics.store(ch.hdr, RPC_CLOSED, conn);
  for (const i of [RPC_CONN, ...RPC_RING_IDX]) Atomics.notify(ch.hdr, i);
};

// pthread workers get the channels as their first message, see the Worker wrapper in llama-cpp.js
self.addEventListener('message', (e) => {
  if (e.data?.cmd === 'wllamaRpc') {
    rpcSetChannels(e.data.channels);
    e.stopImmediatePropagation();
  }
});
