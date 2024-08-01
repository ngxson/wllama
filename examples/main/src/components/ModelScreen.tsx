import { ModelState } from "../utils/types";
import { useWllama } from "../utils/wllama.context";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrashAlt } from '@fortawesome/free-solid-svg-icons';
import { DEFAULT_INFERENCE_PARAMS } from "../utils/config";

export default function ModelScreen() {
  const {
    models,
    downloadModel,
    removeModel,
    loadModel,
    unloadModel,
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

  return <div className="w-[40rem] max-w-full h-full px-4 overflow-auto">
    <div className="inference-params pt-8">
      <h1 className="text-2xl mb-4">
        Inference parameters
      </h1>
      <label className="input input-bordered flex items-center gap-2 mb-2">
        # threads
        <input type="number" className="grow" placeholder="(auto detected)" min="1" max="100" step="1" onChange={onChange('nThreads')} value={currParams.nThreads < 1 ? '' : currParams.nThreads} disabled={blockModelBtn} />
      </label>

      <label className="input input-bordered flex items-center gap-2 mb-2">
        Context size
        <input type="number" className="grow" min="128" step="1" onChange={onChange('nContext')} value={currParams.nContext} disabled={blockModelBtn} />
      </label>

      <label className="input input-bordered flex items-center gap-2 mb-2">
        Max generated tokens
        <input type="number" className="grow" min="10" step="1" onChange={onChange('nPredict')} value={currParams.nPredict} />
      </label>

      <label className="input input-bordered flex items-center gap-2 mb-2">
        Temperature
        <input type="number" className="grow" min="0.0" step="0.05" onChange={onChange('temperature')} value={currParams.temperature} />
      </label>

      <button className="btn btn-sm mr-2" onClick={() => setParams(DEFAULT_INFERENCE_PARAMS)}>Reset params</button>
      <button className="btn btn-sm mr-2" onClick={() => {
        if (confirm('This will remove all downloaded model files from cache. Continue?')) {
          models.forEach(m => removeModel(m));
        }
      }} disabled={blockModelBtn}>Clear cache</button>
    </div>

    <div className="model-management">
      <h1 className="text-2xl mt-6 mb-4">Model management</h1>

      {models.map(m => {
        const percent = parseInt(Math.round(m.downloadPercent * 100).toString());
        return <div className={`card bg-base-100 w-full mb-2 ${m.state === ModelState.LOADED ? 'border-2 border-primary' : ''}`} key={m.url}>
          <div className="card-body p-4 flex flex-row">
            <div className="grow">
              <b>{m.url.split('/').pop()}</b><br />
              <small>
                Size: {m.size} {m.state == ModelState.DOWNLOADING ? ` - Downloaded: ${percent}%` : ''}
              </small>

              {m.state === ModelState.DOWNLOADING && <div>
                <progress className="progress progress-primary w-full" value={percent} max="100"></progress>
              </div>}

              {m.state === ModelState.LOADING && <div>
                <progress className="progress progress-primary w-full"></progress>
              </div>}
            </div>
            <div>
              {m.state === ModelState.NOT_DOWNLOADED && (
                <button className="btn btn-primary btn-sm mr-2" onClick={() => downloadModel(m)} disabled={blockModelBtn}>
                  Download
                </button>
              )}
              {m.state === ModelState.READY && <>
                <button className="btn btn-primary btn-sm mr-2" onClick={() => loadModel(m)} disabled={blockModelBtn}>
                  Load model
                </button>
                <button className="btn btn-outline btn-error btn-sm mr-2" onClick={() => {
                  if (confirm(`Are you sure to remove this model from cache?`)) {
                    removeModel(m);
                  }
                }} disabled={blockModelBtn}>
                  <FontAwesomeIcon icon={faTrashAlt} />
                </button>
              </>}
              {m.state === ModelState.LOADED && (
                <button className="btn btn-outline btn-primary btn-sm mr-2" onClick={() => unloadModel()}>
                  Unload
                </button>
              )}
            </div>
          </div>
        </div>
      })}
    </div>
  </div>;
}
