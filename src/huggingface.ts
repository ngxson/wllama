import { type ModelSource } from './model-manager';

export interface HuggingFaceParams {
  /**
   * The repo name, e.g. user/model
   */
  repo: string;
  /**
   * The file name or path to file in the repo. Only file or quant is needed.
   */
  file?: string;
  /**
   * The GGUF quantization name, e.g. Q4_K_M, Q8_0, etc. Only file or quant is needed.
   *
   * By default, Q4_K_M will be used, then fallback to Q8_0, and finally the non-quantized version if no quantized version is found.
   */
  quant?: string;
  /**
   * The file name or path to file in the repo for mmproj. Only mmprojFile or mmprojQuant is needed.
   */
  mmprojFile?: string;
  /**
   * The GGUF quantization name for mmproj, e.g. Q4_K_M, Q8_0, etc. Only mmprojFile or mmprojQuant is needed.
   */
  mmprojQuant?: string;
  /**
   * The Hugging Face token with permission to access the repo. It can be omitted if the repo is public.
   */
  hfToken?: string;
}

const HF_BASE = 'https://huggingface.co';
const DEFAULT_QUANTS = ['Q4_K_M', 'Q8_0'];

interface HFFileEntry {
  type: string;
  path: string;
  size: number;
  oid: string;
  lfs?: { oid: string; size: number };
}

async function fetchRepoFiles(
  repo: string,
  token?: string
): Promise<HFFileEntry[]> {
  const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
  const headers: Record<string, string> = { Accept: 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const res = await fetch(url, { headers });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      msg = (await res.json()).error ?? msg;
    } catch {
      /* ignore */
    }
    throw new Error(`HF API error (${res.status}): ${msg}`);
  }
  return res.json();
}

// For split GGUF (-00001-of-00005.gguf), return the first shard path.
// For non-split, return path unchanged.
function firstShardPath(files: HFFileEntry[], path: string): string {
  const m = path.match(/^(.+)-(\d{5})-of-(\d{5})\.gguf$/i);
  if (!m) return path;
  const first = `${m[1]}-00001-of-${m[3]}.gguf`;
  return files.some((f) => f.path === first) ? first : path;
}

function selectFile(
  files: HFFileEntry[],
  quant: string | undefined,
  mmprojOnly: boolean
): string | null {
  const candidates = files.filter((f) => {
    if (f.type !== 'file' || !f.path.toLowerCase().endsWith('.gguf'))
      return false;
    const ismmproj = f.path.toLowerCase().includes('mmproj');
    return mmprojOnly ? ismmproj : !ismmproj;
  });

  if (candidates.length === 0) return null;

  if (quant) {
    const upper = quant.toUpperCase();
    const match = candidates.find((f) => f.path.toUpperCase().includes(upper));
    if (match) return firstShardPath(candidates, match.path);
    return null;
  }

  for (const q of DEFAULT_QUANTS) {
    const match = candidates.find((f) => f.path.toUpperCase().includes(q));
    if (match) return firstShardPath(candidates, match.path);
  }

  // Fallback: first candidate
  return firstShardPath(candidates, candidates[0].path);
}

export async function getHFModelSource(
  config: HuggingFaceParams
): Promise<ModelSource> {
  const { repo, file, quant, mmprojFile, mmprojQuant, hfToken } = config;

  const files = await fetchRepoFiles(repo, hfToken);

  const modelPath = file ?? selectFile(files, quant, false);
  if (!modelPath) {
    throw new Error(`No GGUF file found in repo "${repo}"`);
  }

  const source: ModelSource = {
    url: `${HF_BASE}/${repo}/resolve/main/${modelPath}`,
  };

  if (mmprojFile || mmprojQuant !== undefined) {
    const mmpath = mmprojFile ?? selectFile(files, mmprojQuant, true);
    if (mmpath) {
      source.mmprojUrl = `${HF_BASE}/${repo}/resolve/main/${mmpath}`;
    }
  }

  if (hfToken) {
    const params = new URLSearchParams({ token: hfToken });
    source.url += `?${params}`;
    if (source.mmprojUrl) {
      source.mmprojUrl += `?${params}`;
    }
  }

  return source;
}
