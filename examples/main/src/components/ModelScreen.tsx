import { ManageModel, ModelState } from '../utils/types';
import { useWllama } from '../utils/wllama.context';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faTrashAlt,
  faXmark,
  faWarning,
} from '@fortawesome/free-solid-svg-icons';
import { DEFAULT_INFERENCE_PARAMS, MAX_GGUF_SIZE } from '../config';
import { toHumanReadableSize } from '../utils/utils';
import { useState } from 'react';

export default function ModelScreen() {
  const [showAddCustom, setShowAddCustom] = useState(false);
  const {
    models,
    removeModel,
    isLoadingModel,
    isDownloading,
    currModel,
    currParams,
    setParams,
  } = useWllama();

  const blockModelBtn = !!(currModel || isDownloading || isLoadingModel);

  const onChange = (key: keyof typeof currParams) => (e: any) => {
    setParams({ ...currParams, [key]: parseFloat(e.target.value || -1) });
  };

  return (
    <div className="w-[40rem] max-w-full h-full px-4 overflow-auto">
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
                await removeModel(m);
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
            + Add model
          </button>
        </h1>

        {models
          .filter((m) => m.userAdded)
          .map((m) => (
            <ModelCard key={m.url} model={m} blockModelBtn={blockModelBtn} />
          ))}
      </div>

      <div className="model-management">
        <h1 className="text-2xl mt-6 mb-4">Recommended models</h1>

        {models
          .filter((m) => !m.userAdded)
          .map((m) => (
            <ModelCard key={m.url} model={m} blockModelBtn={blockModelBtn} />
          ))}
      </div>

      <div className="h-10" />

      {showAddCustom && (
        <AddCustomModelDialog onClose={() => setShowAddCustom(false)} />
      )}
    </div>
  );
}

function AddCustomModelDialog({ onClose }: { onClose(): void }) {
  const { isLoadingModel, addCustomModel } = useWllama();
  const [url, setUrl] = useState<string>('');
  const [err, setErr] = useState<string>();

  const onSubmit = async () => {
    try {
      await addCustomModel(url);
      onClose();
    } catch (e) {
      setErr((e as any)?.message ?? 'unknown error');
    }
  };

  return (
    <dialog className="modal modal-open">
      <div className="modal-box">
        <h3 className="font-bold text-lg">Add custom model</h3>
        <div className="mt-4">
          Max gguf file size is 2GB. If your model is bigger than 2GB, please{' '}
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
            URL
            <input
              type="text"
              className="grow"
              placeholder="https://example.com/your_model-00001-of-00XXX.gguf"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
          </label>
        </div>
        {err && <div className="mt-4 text-error">Error: {err}</div>}
        <div className="modal-action">
          <button
            className="btn btn-primary"
            disabled={isLoadingModel || url.length < 5}
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
  model: ManageModel;
  blockModelBtn: boolean;
}) {
  const {
    downloadModel,
    removeModel,
    loadModel,
    unloadModel,
    removeCustomModel,
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
          <b>{m.name}</b>
          <br />
          <small>
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
                    removeModel(m);
                  }
                }}
                disabled={blockModelBtn}
              >
                <FontAwesomeIcon icon={faTrashAlt} />
              </button>
            </>
          )}
          {m.state === ModelState.LOADED && (
            <button
              className="btn btn-outline btn-primary btn-sm mr-2"
              onClick={() => unloadModel()}
            >
              Unload
            </button>
          )}
          {m.state === ModelState.NOT_DOWNLOADED && m.userAdded && (
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
