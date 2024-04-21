export declare const joinBuffers: (buffers: Uint8Array[]) => Uint8Array;
/**
 * Load a resource as byte array. If multiple URLs is given, we will assume that the resource is splitted into small files
 * @param url URL (or list of URLs) to resource
 */
export declare const loadBinaryResource: (url: string | string[], nMaxParallel: number) => Promise<Uint8Array | Uint8Array[]>;
/**
 * Convert list of bytes (number) to text
 * @param buffer
 * @returns a string
 */
export declare const bufToText: (buffer: Uint8Array) => string;
/**
 * Get default stdout/stderr config for wasm module
 */
export declare const getWModuleConfig: (pathConfig: {
    [filename: string]: string;
}) => {
    noInitialRun: boolean;
    print: (text: any) => void;
    printErr: (text: any) => void;
    locateFile: (filename: string, basePath: string) => string;
};
export declare const delay: (ms: number) => Promise<unknown>;
export declare const absoluteUrl: (relativePath: string) => string;
export declare const padDigits: (number: number, digits: number) => string;
/**
 * Browser feature detection
 * Copied from https://unpkg.com/wasm-feature-detect?module (Apache License)
 */
/**
 * @returns true if browser support multi-threads
 */
export declare const isSupportMultiThread: () => Promise<boolean>;
/**
 * Throws an error if the environment is not compatible
 */
export declare const checkEnvironmentCompatible: () => Promise<void>;
