import { useEffect, useMemo, useState } from 'react';
import { Wllama } from '@wllama/wllama';

// See: https://vitejs.dev/guide/assets#explicit-url-imports
import wllamaSingle from '@wllama/wllama/src/single-thread/wllama.wasm?url';
import wllamaMulti from '@wllama/wllama/src/multi-thread/wllama.wasm?url';

const CONFIG_PATHS = {
  'single-thread/wllama.wasm': wllamaSingle,
  'multi-thread/wllama.wasm': wllamaMulti,
};
const CMPL_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';

function App() {
  const wllama = useMemo(() => new Wllama(CONFIG_PATHS), []);
  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [prompt, setPrompt] = useState('Once upon a time,');
  const [nPredict, setNPredict] = useState('50');
  const [output, setOutput] = useState('');

  useEffect(() => {
    (async () => {
      await wllama.loadModelFromUrl(CMPL_MODEL);
      setReady(true);
    })();
  }, [wllama]);

  const runCompletion = async () => {
    setRunning(true);
    setOutput('');
    await wllama.createCompletion(prompt, {
      nPredict: parseInt(nPredict),
      sampling: {
        temp: 0.5,
        top_k: 40,
        top_p: 0.9,
      },
      // @ts-ignore
      onNewToken: (token, piece, currentText) => {
        setOutput(currentText);
      },
    });
    setRunning(false);
  };

  // For embeddings, see "examples/basic/index.html"

  return (
    <div className="main">
      <h2>Completions demo - ReactJS</h2>
      Model: {CMPL_MODEL} <br />
      {!ready ? (
        <>Loading model...</>
      ) : (
        <>
          Prompt:{' '}
          <input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={running}
          />
          <br />
          Number of tokens:{' '}
          <input
            value={nPredict}
            onChange={(e) => setNPredict(e.target.value)}
            disabled={running}
          />
          <br />
          <button onClick={runCompletion} disabled={running}>
            Run completions
          </button>
          <br />
          <br />
          Completion: <br />
          <div className="output_cmpl">{output}</div>
        </>
      )}
    </div>
  );
}

export default App;
