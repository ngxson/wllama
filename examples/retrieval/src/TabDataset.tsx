import { Button, Card, Form, ProgressBar } from 'react-bootstrap';
import { DataEntry, EMBD_MODEL_NAME, downloadStringAsFile, useAppContext } from './utils';
import { useState } from 'react';
import { Document, CharacterTextSplitter } from './text_splitter';

export default function TabDataset() {
  const { wllama, dbSet, dataset } = useAppContext();
  const [mode, setMode] = useState<'view' | 'input' | 'split' | 'process'>('view');
  const [progress, setProgress] = useState(0);
  const [isDone, setIsDone] = useState(false);
  const [inputText, setInputText] = useState('');
  const [inputChunkSize, setInputChunkSize] = useState('1000');
  const [inputChunkOverlap, setInputChunkOverlap] = useState('200');
  const [inputSplitSeparator, setInputSplitSeparator] = useState('.');
  const [inputPrefix, setInputPrefix] = useState('search_document:');
  const [chunks, setChunks] = useState<Document[]>([]);

  const reset = () => {
    setMode('view');
    setInputText('');
    setChunks([]);
    setProgress(0);
    setIsDone(false);
  };

  const onSubmitInputDoc = (e: React.FormEvent<Element>) => {
    e.preventDefault();
    (async () => {
      const textSplitter = new CharacterTextSplitter({
        chunkSize: parseInt(inputChunkSize),
        chunkOverlap: parseInt(inputChunkOverlap),
        separator: inputSplitSeparator.replace(/\\n/g, '\n'),
      });
      const docs = await textSplitter.createDocuments([inputText]);
      setChunks(docs);
      setMode('split');
    })();
  };

  const onRunEmbeddings = async () => {
    const confirmation = window.confirm('This will erase old dataset, continue?');
    if (confirmation) {
      const dataset: DataEntry[] = [];
      setMode('process');
      setProgress(0);
      for (const chunk of chunks) {
        const prompt = `${inputPrefix.trim()} ${chunk.pageContent}`.trim();
        const embd = await wllama.createEmbedding(prompt);
        // console.log(embd)
        dataset.push({
          embeddings: embd,
          content: chunk.pageContent,
          metadata: chunk.metadata.loc,
        });
        setProgress(n => n + 1);
      }
      await dbSet(dataset);
      setIsDone(true);
    }
  };

  // TODO: This function is soly for exporting demo dataset. To expose it to frontend, we need to implement also "Import" function
  // @ts-ignore
  window.__exportDataset = () => {
    const jsonLines = dataset.map(d => JSON.stringify(d));
    const jsonOutput = `[\n${jsonLines.join(',\n')}\n]`;
    downloadStringAsFile('dataset.json', jsonOutput);
  };

  return <>
    <Card className='mb-4'>
      {mode === 'view' && <Card.Body>
        <b>Model</b>: {EMBD_MODEL_NAME} <br/>
        <br/>
        <Button onClick={() => setMode('input')}>Import your own dataset</Button>
        &nbsp;&nbsp;
        <Button>Import demo dataset</Button>
      </Card.Body>}

      {mode === 'input' && <Card.Body>
        <b>Model</b>: {EMBD_MODEL_NAME} <br/>
        <b>Split strategy</b>: CharacterTextSplitter <br/>
        <Form onSubmit={onSubmitInputDoc}>
          <Form.Group>
            <Form.Label>Chunk size (number of characters):</Form.Label>
            <Form.Control type='number' value={inputChunkSize} onChange={e => setInputChunkSize(e.target.value)} />
          </Form.Group>
          <Form.Group>
            <Form.Label>Chunk overlap (number of characters):</Form.Label>
            <Form.Control type='number' value={inputChunkOverlap} onChange={e => setInputChunkOverlap(e.target.value)} />
          </Form.Group>
          <Form.Group>
            <Form.Label>Split sequence (for example: <code>.</code> or <code>\n\n</code>):</Form.Label>
            <Form.Control value={inputSplitSeparator} onChange={e => setInputSplitSeparator(e.target.value)} />
          </Form.Group>
          <Form.Group>
            <Form.Label>Prompt prefix (<a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1#usage" target='_blank' rel="noopener">What is this?</a>):</Form.Label>
            <Form.Control value={inputPrefix} onChange={e => setInputPrefix(e.target.value)} />
          </Form.Group>
          <Form.Group>
            <Form.Label>Paste your document here:</Form.Label>
            <Form.Control
              as='textarea'
              placeholder=''
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              style={{ height: '30em' }}
            />
          </Form.Group>
          <br/>
          <Button type='submit'>Break into chunks</Button>
          &nbsp;&nbsp;
          <Button type='button' variant='secondary' onClick={reset}>Cancel</Button>
        </Form>
      </Card.Body>}

      {mode === 'split' && <Card.Body>
        <b>Model</b>: {EMBD_MODEL_NAME} <br/>
        Here is the document broken into smaller chunks. You can review them before calculating the embeddings and import into database. <br/>
        Total number of chunks: {chunks.length}<br/>
        <br/>
        <Button type='submit' onClick={onRunEmbeddings}>Calculate embeddings and import</Button>
        &nbsp;&nbsp;
        <Button variant='secondary' onClick={reset}>Cancel</Button>
      </Card.Body>}

      {mode === 'process' && <Card.Body>
        <b>Model</b>: {EMBD_MODEL_NAME} <br/>
        {!isDone ? `Calculate embeddings... [${progress} / ${chunks.length}]` : 'Done!'} <br/>
        <ProgressBar
          animated={!isDone}
          now={progress / chunks.length * 100}
        />
        <br/>
        {isDone && <Button onClick={reset}>Go back</Button>}
      </Card.Body>}
    </Card>

    {mode === 'split' && chunks.map((chunk, i) => {
      return <Card key={i} className='mb-2'>
        <Card.Body>
          <b>Chunk {i+1} | Metadata: {JSON.stringify(chunk.metadata.loc)}</b><br/>
          {chunk.pageContent}
        </Card.Body>
      </Card>
    })}

    {mode === 'view' && <>
      <h5>Chunks in dataset:</h5>
      {dataset.map((chunk, i) => {
        return <Card key={i} className='mb-2'>
          <Card.Body>
            {chunk.content}
          </Card.Body>
        </Card>
      })}
    </>}
  </>
}