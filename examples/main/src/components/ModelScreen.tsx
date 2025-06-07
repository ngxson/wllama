import { ModelState, Screen } from '../utils/types';
import { useWllama } from '../utils/wllama.context';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faTrashAlt,
  faXmark,
  faWarning,
  faCheck,
} from '@fortawesome/free-solid-svg-icons';
import { DEFAULT_INFERENCE_PARAMS, MAX_GGUF_SIZE } from '../config';
import { toHumanReadableSize, useDebounce } from '../utils/utils';
import { useEffect, useState } from 'react';
import ScreenWrapper from './ScreenWrapper';
import { DisplayedModel } from '../utils/displayed-model';
import { isValidGgufFile } from '@wllama/wllama';

export default function ModelScreen() {
  const [showAddCustom, setShowAddCustom] = useState(false);
  const {
    models,
    removeCachedModel,
    isLoadingModel,
    isDownloading,
    loadedModel,
    currParams,
    setParams,
  } = useWllama();

  const blockModelBtn = !!(loadedModel || isDownloading || isLoadingModel);

  const onChange = (key: keyof typeof currParams) => (e: any) => {
    setParams({ ...currParams, [key]: parseFloat(e.target.value || -1) });
  };

  return (
    <ScreenWrapper>
      <div className="inference-params pt-8">
        <h1 className="text-2xl mb-4">Inference parameters</h1>
        <label className="input input-bordered flex items-center gap-2 mb-2">
          # threads
          <input
            type="number"
            className="grow"
            placeholder="(auto detected)"
            min="1"
            max="100"
            step="1"
            onChange={onChange('nThreads')}
            value={currParams.nThreads < 1 ? '' : currParams.nThreads}
            disabled={blockModelBtn}
          />
        </label>

        <label className="input input-bordered flex items-center gap-2 mb-2">
          Context size
          <input
            type="number"
            className="grow"
            min="128"
            step="1"
            onChange={onChange('nContext')}
            value={currParams.nContext}
            disabled={blockModelBtn}
          />
        </label>

        <label className="input input-bordered flex items-center gap-2 mb-2">
          Max generated tokens
          <input
            type="number"
            className="grow"
            min="10"
            step="1"
            onChange={onChange('nPredict')}
            value={currParams.nPredict}
          />
        </label>

        <label className="input input-bordered flex items-center gap-2 mb-2">
          Temperature
          <input
            type="number"
            className="grow"
            min="0.0"
            step="0.05"
            onChange={onChange('temperature')}
            value={currParams.temperature}
          />
        </label>

        <button
          className="btn btn-sm mr-2"
          onClick={() => setParams(DEFAULT_INFERENCE_PARAMS)}
        >
          Reset params
        </button>
        <button
          className="btn btn-sm mr-2"
          onClick={async () => {
            if (
              confirm(
                'This will remove all downloaded model files from cache. Continue?'
              )
            ) {
              for (const m of models) {
                await removeCachedModel(m);
              }
            }
          }}
          disabled={blockModelBtn}
        >
          Clear cache
        </button>
      </div>

      <div className="model-management">
        <h1 className="text-2xl mt-6 mb-4">
          Custom models
          <button
            className="btn btn-primary btn-outline btn-sm ml-6"
            onClick={() => setShowAddCustom(true)}
          >
            + Add GGUF
          </button>
        </h1>

        {models
          .filter((m) => m.isUserAdded)
          .map((m) => (
            <ModelCard key={m.url} model={m} blockModelBtn={blockModelBtn} />
          ))}
      </div>

      <div className="model-management">
        <h1 className="text-2xl mt-6 mb-4">Recommended models</h1>

        {models
          .filter((m) => !m.isUserAdded)
          .map((m) => (
            <ModelCard key={m.url} model={m} blockModelBtn={blockModelBtn} />
          ))}
      </div>

      <div className="h-10" />

      {showAddCustom && (
        <AddCustomModelDialog onClose={() => setShowAddCustom(false)} />
      )}
    </ScreenWrapper>
  );
}

function AddCustomModelDialog({ onClose }: { onClose(): void }) {
  const { isLoadingModel, addCustomModel } = useWllama();
  const [hfRepo, setHfRepo] = useState<string>('');
  const [hfFile, setHfFile] = useState<string>('');
  const [hfFiles, setHfFiles] = useState<string[]>([]);
  const [abortSignal, setAbortSignal] = useState<AbortController>(
    new AbortController()
  );
  const [err, setErr] = useState<string>();

  useDebounce(
    async () => {
      if (hfRepo.length < 2) {
        setHfFiles([]);
        return;
      }
      try {
        const res = await fetch(`https://huggingface.co/api/models/${hfRepo}`, {
          signal: abortSignal.signal,
        });
        const data: { siblings?: { rfilename: string }[] } = await res.json();
        if (data.siblings) {
          setHfFiles(
            data.siblings
              .map((s) => s.rfilename)
              .filter((f) => isValidGgufFile(f))
          );
          setErr('');
        } else {
          setErr('no model found or it is private');
          setHfFiles([]);
        }
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
          setErr((e as any)?.message ?? 'unknown error');
          setHfFiles([]);
        }
      }
    },
    [hfRepo],
    500
  );

  useEffect(() => {
    if (hfFiles.length === 0) {
      setHfFile('');
    }
  }, [hfFiles]);

  const onSubmit = async () => {
    try {
      await addCustomModel(
        `https://huggingface.co/${hfRepo}/resolve/main/${hfFile}`
      );
      onClose();
    } catch (e) {
      setErr((e as any)?.message ?? 'unknown error');
    }
  };

  return (
    <dialog className="modal modal-open">
      <div className="modal-box">
        <h3 className="font-bold text-lg">Add custom GGUF</h3>
        <div className="mt-4">
          Max GGUF file size is 2GB. If your model is bigger than 2GB, please{' '}
          <a
            href="https://github.com/ngxson/wllama?tab=readme-ov-file#split-model"
            target="_blank"
            rel="noopener"
            className="text-primary"
          >
            follow this guide
          </a>{' '}
          to split it into smaller shards.
        </div>
        <div className="mt-4">
          <label className="input input-bordered flex items-center gap-2 mb-2">
            HF repo
            <input
              type="text"
              className="grow"
              placeholder="{username}/{repo}"
              value={hfRepo}
              onChange={(e) => {
                abortSignal.abort();
                setHfRepo(e.target.value);
                setAbortSignal(new AbortController());
              }}
            />
          </label>
          <select
            className="select select-bordered w-full"
            onChange={(e) => setHfFile(e.target.value)}
          >
            <option value="">Select a model file</option>
            {hfFiles.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>
        {err && <div className="mt-4 text-error">Error: {err}</div>}
        <div className="modal-action">
          <button
            className="btn btn-primary"
            disabled={isLoadingModel || hfRepo.length < 2 || hfFile.length < 5}
            onClick={onSubmit}
          >
            {isLoadingModel && (
              <span className="loading loading-spinner"></span>
            )}
            Add model
          </button>
          <button className="btn" disabled={isLoadingModel} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </dialog>
  );
}

function ModelCard({
  model,
  blockModelBtn,
}: {
  model: DisplayedModel;
  blockModelBtn: boolean;
}) {
  const {
    downloadModel,
    removeCachedModel,
    loadModel,
    unloadModel,
    removeCustomModel,
    currRuntimeInfo,
    navigateTo,
  } = useWllama();

  const m = model;
  const percent = parseInt(Math.round(m.downloadPercent * 100).toString());
  return (
    <div
      className={`card bg-base-100 w-full mb-2 ${m.state === ModelState.LOADED ? 'border-2 border-primary' : ''}`}
      key={m.url}
    >
      <div className="card-body p-4 flex flex-row">
        <div className="grow">
          <b>{m.hfPath.replace(/-\d{5}-of-\d{5}/, '-(shards)')}</b>
          <br />
          <small>
            HF repo: {m.hfModel}
            <br />
            Size: {toHumanReadableSize(m.size)}
            {m.size > MAX_GGUF_SIZE && (
              <div
                className="tooltip tooltip-right"
                data-tip="Big model size, may not be able to load due to RAM limitation"
              >
                <span className="text-yellow-300 ml-2">
                  <FontAwesomeIcon icon={faWarning} />
                </span>
              </div>
            )}
            {m.state == ModelState.DOWNLOADING
              ? ` - Downloaded: ${percent}%`
              : ''}
          </small>

          {m.state === ModelState.LOADED && currRuntimeInfo && (
            <>
              <br />
              <InfoOnOffDisplay
                text="Multithread"
                on={currRuntimeInfo.isMultithread}
              />
              &nbsp;&nbsp;&nbsp;&nbsp;
              <InfoOnOffDisplay
                text="Chat template"
                on={currRuntimeInfo.hasChatTemplate}
              />
            </>
          )}

          {m.state === ModelState.DOWNLOADING && (
            <div>
              <progress
                className="progress progress-primary w-full"
                value={percent}
                max="100"
              ></progress>
            </div>
          )}

          {m.state === ModelState.LOADING && (
            <div>
              <progress className="progress progress-primary w-full"></progress>
            </div>
          )}
        </div>
        <div>
          {m.state === ModelState.NOT_DOWNLOADED && (
            <button
              className="btn btn-primary btn-sm mr-2"
              onClick={() => downloadModel(m)}
              disabled={blockModelBtn}
            >
              Download
            </button>
          )}
          {m.state === ModelState.READY && (
            <>
              <button
                className="btn btn-primary btn-sm mr-2"
                onClick={() => loadModel(m)}
                disabled={blockModelBtn}
              >
                Load model
              </button>
              <button
                className="btn btn-outline btn-error btn-sm mr-2"
                onClick={() => {
                  if (
                    confirm('Are you sure to remove this model from cache?')
                  ) {
                    removeCachedModel(m);
                  }
                }}
                disabled={blockModelBtn}
              >
                <FontAwesomeIcon icon={faTrashAlt} />
              </button>
            </>
          )}
          {m.state === ModelState.LOADED && (
            <>
              <button
                className="btn btn-primary btn-sm mr-2"
                onClick={() => navigateTo(Screen.CHAT)}
              >
                Start chat
              </button>
              <button
                className="btn btn-outline btn-primary btn-sm mr-2"
                onClick={() => unloadModel()}
              >
                Unload
              </button>
            </>
          )}
          {m.state === ModelState.NOT_DOWNLOADED && m.isUserAdded && (
            <button
              className="btn btn-outline btn-error btn-sm mr-2"
              onClick={() => {
                if (
                  confirm('Are you sure to remove this model from the list?')
                ) {
                  removeCustomModel(m);
                }
              }}
              disabled={blockModelBtn}
            >
              <FontAwesomeIcon icon={faXmark} />
            </button>
          )}
          {m.state == ModelState.DOWNLOADING && (
            <span className="loading loading-spinner"></span>
          )}
        </div>
      </div>
    </div>
  );
}

function InfoOnOffDisplay({ text, on }: { text: string; on: boolean }) {
  return (
    <>
      {on ? (
        <span className="text-green-300">
          <FontAwesomeIcon icon={faCheck} />
        </span>
      ) : (
        <span className="text-red-400">
          <FontAwesomeIcon icon={faXmark} />
        </span>
      )}
      <span className="text-sm">&nbsp;{text}</span>
    </>
  );
}
