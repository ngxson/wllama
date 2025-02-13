// Start the main llama.cpp
let wllamaMalloc;
let wllamaStart;
let wllamaAction;
let wllamaExit;
let wllamaDebug;

let Module = null;

//////////////////////////////////////////////////////////////
// UTILS
//////////////////////////////////////////////////////////////

// send message back to main thread
const msg = (data, transfer) => postMessage(data, transfer);

// Convert CPP log into JS log
const cppLogToJSLog = (line) => {
  const matched = line.match(/@@(DEBUG|INFO|WARN|ERROR)@@(.*)/);
  return !!matched
    ? {
        level: (matched[1] === 'INFO' ? 'debug' : matched[1]).toLowerCase(),
        text: matched[2],
      }
    : { level: 'log', text: line };
};

// Get module config that forwards stdout/err to main thread
const getWModuleConfig = (_argMainScriptBlob) => {
  var pathConfig = RUN_OPTIONS.pathConfig;
  var pthreadPoolSize = RUN_OPTIONS.nbThread;
  var argMainScriptBlob = _argMainScriptBlob;

  if (!pathConfig['wllama.wasm']) {
    throw new Error('"wllama.wasm" is missing in pathConfig');
  }
  return {
    noInitialRun: true,
    print: function (text) {
      if (arguments.length > 1)
        text = Array.prototype.slice.call(arguments).join(' ');
      msg({ verb: 'console.log', args: [text] });
    },
    printErr: function (text) {
      if (arguments.length > 1)
        text = Array.prototype.slice.call(arguments).join(' ');
      const logLine = cppLogToJSLog(text);
      msg({ verb: 'console.' + logLine.level, args: [logLine.text] });
    },
    locateFile: function (filename, basePath) {
      const p = pathConfig[filename];
      const truncate = (str) =>
        str.length > 128 ? `${str.substr(0, 128)}...` : str;
      if (filename.match(/wllama\.worker\.js/)) {
        msg({
          verb: 'console.error',
          args: [
            '"wllama.worker.js" is removed from v2.2.1. Hint: make sure to clear browser\'s cache.',
          ],
        });
      } else {
        msg({
          verb: 'console.debug',
          args: [`Loading "${filename}" from "${truncate(p)}"`],
        });
        return p;
      }
    },
    mainScriptUrlOrBlob: argMainScriptBlob,
    pthreadPoolSize,
    wasmMemory: pthreadPoolSize > 1 ? getWasmMemory() : null,
    onAbort: function (text) {
      msg({ verb: 'signal.abort', args: [text] });
    },
  };
};

// Get the memory to be used by wasm. (Only used in multi-thread mode)
// Because we have a weird OOM issue on iOS, we need to try some values
// See: https://github.com/emscripten-core/emscripten/issues/19144
//      https://github.com/godotengine/godot/issues/70621
const getWasmMemory = () => {
  let minBytes = 128 * 1024 * 1024;
  let maxBytes = 4096 * 1024 * 1024;
  let stepBytes = 128 * 1024 * 1024;
  while (maxBytes > minBytes) {
    try {
      const wasmMemory = new WebAssembly.Memory({
        initial: minBytes / 65536,
        maximum: maxBytes / 65536,
        shared: true,
      });
      return wasmMemory;
    } catch (e) {
      maxBytes -= stepBytes;
      continue; // retry
    }
  }
  throw new Error('Cannot allocate WebAssembly.Memory');
};

//////////////////////////////////////////////////////////////
// MEMFS PATCH
//////////////////////////////////////////////////////////////

/**
 * By default, emscripten uses memfs. The way it works is by
 * allocating new Uint8Array in javascript heap. This is not good
 * because it requires files to be copied to wasm heap each time
 * a file is read.
 *
 * HeapFS is an alternative, which resolves this problem by
 * allocating space for file directly inside wasm heap. This
 * allows us to mmap without doing any copy.
 *
 * For llama.cpp, this is great because we use MAP_SHARED
 *
 * Ref: https://github.com/ngxson/wllama/pull/39
 * Ref: https://github.com/emscripten-core/emscripten/blob/main/src/library_memfs.js
 *
 * Note 29/05/2024 @ngxson
 * Due to ftell() being limited to MAX_LONG, we cannot load files bigger than 2^31 bytes (or 2GB)
 * Ref: https://github.com/emscripten-core/emscripten/blob/main/system/lib/libc/musl/src/stdio/ftell.c
 */

const fsNameToFile = {}; // map Name => File
const fsIdToFile = {}; // map ID => File
let currFileId = 0;

// Patch and redirect memfs calls to wllama
const patchMEMFS = () => {
  const m = Module;
  // save functions
  m.MEMFS.stream_ops._read = m.MEMFS.stream_ops.read;
  m.MEMFS.stream_ops._write = m.MEMFS.stream_ops.write;
  m.MEMFS.stream_ops._llseek = m.MEMFS.stream_ops.llseek;
  m.MEMFS.stream_ops._allocate = m.MEMFS.stream_ops.allocate;
  m.MEMFS.stream_ops._mmap = m.MEMFS.stream_ops.mmap;
  m.MEMFS.stream_ops._msync = m.MEMFS.stream_ops.msync;

  const patchStream = (stream) => {
    const name = stream.node.name;
    if (fsNameToFile[name]) {
      const f = fsNameToFile[name];
      stream.node.contents = m.HEAPU8.subarray(f.ptr, f.ptr + f.size);
      stream.node.usedBytes = f.size;
    }
  };

  // replace "read" functions
  m.MEMFS.stream_ops.read = function (
    stream,
    buffer,
    offset,
    length,
    position
  ) {
    patchStream(stream);
    return m.MEMFS.stream_ops._read(stream, buffer, offset, length, position);
  };
  m.MEMFS.ops_table.file.stream.read = m.MEMFS.stream_ops.read;

  // replace "llseek" functions
  m.MEMFS.stream_ops.llseek = function (stream, offset, whence) {
    patchStream(stream);
    return m.MEMFS.stream_ops._llseek(stream, offset, whence);
  };
  m.MEMFS.ops_table.file.stream.llseek = m.MEMFS.stream_ops.llseek;

  // replace "mmap" functions
  m.MEMFS.stream_ops.mmap = function (stream, length, position, prot, flags) {
    patchStream(stream);
    const name = stream.node.name;
    if (fsNameToFile[name]) {
      const f = fsNameToFile[name];
      return {
        ptr: f.ptr + position,
        allocated: false,
      };
    } else {
      return m.MEMFS.stream_ops._mmap(stream, length, position, prot, flags);
    }
  };
  m.MEMFS.ops_table.file.stream.mmap = m.MEMFS.stream_ops.mmap;

  // mount FS
  m.FS.mkdir('/models');
  m.FS.mount(m.MEMFS, { root: '.' }, '/models');
};

// Allocate a new file in wllama heapfs, returns file ID
const heapfsAlloc = (name, size) => {
  if (size < 1) {
    throw new Error('File size must be bigger than 0');
  }
  const m = Module;
  const ptr = m.mmapAlloc(size);
  const file = {
    ptr: ptr,
    size: size,
    id: currFileId++,
  };
  fsIdToFile[file.id] = file;
  fsNameToFile[name] = file;
  return file.id;
};

// Add new file to wllama heapfs, return number of written bytes
const heapfsWrite = (id, buffer, offset) => {
  const m = Module;
  if (fsIdToFile[id]) {
    const { ptr, size } = fsIdToFile[id];
    const afterWriteByte = offset + buffer.byteLength;
    if (afterWriteByte > size) {
      throw new Error(
        `File ID ${id} write out of bound, afterWriteByte = ${afterWriteByte} while size = ${size}`
      );
    }
    m.HEAPU8.set(buffer, ptr + offset);
    return buffer.byteLength;
  } else {
    throw new Error(`File ID ${id} not found in heapfs`);
  }
};

//////////////////////////////////////////////////////////////
// MAIN CODE
//////////////////////////////////////////////////////////////

const callWrapper = (name, ret, args) => {
  const fn = Module.cwrap(name, ret, args);
  return async (action, req) => {
    let result;
    try {
      if (args.length === 2) {
        result = await fn(action, req);
      } else {
        result = fn();
      }
    } catch (ex) {
      console.error(ex);
      throw ex;
    }
    return result;
  };
};

onmessage = async (e) => {
  if (!e.data) return;
  const { verb, args, callbackId } = e.data;

  if (!callbackId) {
    msg({ verb: 'console.error', args: ['callbackId is required', e.data] });
    return;
  }

  if (verb === 'module.init') {
    const argMainScriptBlob = args[0];
    try {
      Module = getWModuleConfig(argMainScriptBlob);
      Module.onRuntimeInitialized = () => {
        // async call once module is ready
        // init FS
        patchMEMFS();
        // init cwrap
        const pointer = 'number';
        // TODO: note sure why emscripten cannot bind if there is only 1 argument
        wllamaMalloc = callWrapper('wllama_malloc', pointer, [
          'number',
          pointer,
        ]);
        wllamaStart = callWrapper('wllama_start', 'string', []);
        wllamaAction = callWrapper('wllama_action', pointer, [
          'string',
          pointer,
        ]);
        wllamaExit = callWrapper('wllama_exit', 'string', []);
        wllamaDebug = callWrapper('wllama_debug', 'string', []);
        msg({ callbackId, result: null });
      };
      wModuleInit();
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'fs.alloc') {
    const argFilename = args[0];
    const argSize = args[1];
    try {
      // create blank file
      const emptyBuffer = new ArrayBuffer(0);
      Module['FS_createDataFile'](
        '/models',
        argFilename,
        emptyBuffer,
        true,
        true,
        true
      );
      // alloc data on heap
      const fileId = heapfsAlloc(argFilename, argSize);
      msg({ callbackId, result: { fileId } });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'fs.write') {
    const argFileId = args[0];
    const argBuffer = args[1];
    const argOffset = args[2];
    try {
      const writtenBytes = heapfsWrite(argFileId, argBuffer, argOffset);
      msg({ callbackId, result: { writtenBytes } });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.start') {
    try {
      const result = await wllamaStart();
      msg({ callbackId, result });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.action') {
    const argAction = args[0];
    const argEncodedMsg = args[1];
    try {
      const inputPtr = await wllamaMalloc(argEncodedMsg.byteLength, 0);
      // copy data to wasm heap
      const inputBuffer = new Uint8Array(
        Module.HEAPU8.buffer,
        inputPtr,
        argEncodedMsg.byteLength
      );
      inputBuffer.set(argEncodedMsg, 0);
      const outputPtr = await wllamaAction(argAction, inputPtr);
      // length of output buffer is written at the first 4 bytes of input buffer
      const outputLen = new Uint32Array(Module.HEAPU8.buffer, inputPtr, 1)[0];
      // copy the output buffer to JS heap
      const outputBuffer = new Uint8Array(outputLen);
      const outputSrcView = new Uint8Array(
        Module.HEAPU8.buffer,
        outputPtr,
        outputLen
      );
      outputBuffer.set(outputSrcView, 0); // copy it
      msg({ callbackId, result: outputBuffer }, [outputBuffer.buffer]);
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.exit') {
    try {
      const result = await wllamaExit();
      msg({ callbackId, result });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }

  if (verb === 'wllama.debug') {
    try {
      const result = await wllamaDebug();
      msg({ callbackId, result });
    } catch (err) {
      msg({ callbackId, err });
    }
    return;
  }
};
