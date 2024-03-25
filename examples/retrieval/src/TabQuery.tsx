import { Button, Card, Form } from 'react-bootstrap';
import { DataEntry, EMBD_MODEL_NAME, useAppContext } from './utils';
import { useState } from 'react';

interface QueryResult {
  similarity: number;
  data: DataEntry;
};

const vecDot = (a: number[], b: number[]) => a.reduce((acc, _, i) => acc + a[i]*b[i], 0);

export default function TabQuery() {
  const { wllama, dataset } = useAppContext();
  const [inputPrefix, setInputPrefix] = useState('search_query:');
  const [inputNumRes, setInputNumRes] = useState('10');
  const [inputSearch, setInputSearch] = useState('');
  const [resultTerm, setResultTerm] = useState('');
  const [results, setResults] = useState<QueryResult[]>([]);
  const [busy, setBusy] = useState(false);

  const onSubmit = (e: React.FormEvent<Element>) => {
    e.preventDefault();
    setBusy(true);
    (async () => {
      const prompt = `${inputPrefix.trim()} ${inputSearch.trim()}`;
      const embd = await wllama.createEmbedding(prompt);
      const results: QueryResult[] = dataset.map(d => ({
        similarity: vecDot(embd, d.embeddings),
        data: d,
      }));
      results.sort((a, b) => b.similarity - a.similarity);
      setResultTerm(inputSearch);
      setResults(results);
      setBusy(false);
    })();
  };

  return <>
    <Card className='mb-4'>
      <Card.Body>
        <b>Model</b>: {EMBD_MODEL_NAME} <br/>
        <Form onSubmit={onSubmit}>
          <Form.Group>
            <Form.Label>Number of results:</Form.Label>
            <Form.Control
              type='number'
              value={inputNumRes}
              onChange={e => setInputNumRes(e.target.value)}
              disabled={busy}
            />
          </Form.Group>
          <Form.Group>
            <Form.Label>Query prefix:</Form.Label>
            <Form.Control
              placeholder=''
              value={inputPrefix}
              onChange={(e) => setInputPrefix(e.target.value)}
              disabled={busy}
            />
          </Form.Group>
          <Form.Group>
            <Form.Label>Query:</Form.Label>
            <Form.Control
              as='textarea'
              placeholder='Example: What is ...? When did ... happen?'
              value={inputSearch}
              onChange={(e) => setInputSearch(e.target.value)}
              disabled={busy}
            />
          </Form.Group>
          <Button type='submit' className='mt-2' disabled={busy}>Run query</Button>
        </Form>
      </Card.Body>
    </Card>

    {resultTerm.length > 0 && <>
      <h5>Result for: {resultTerm}</h5>
      {results.slice(0, parseInt(inputNumRes)).map((res, i) => <Card key={i} className='mt-2'>
        <Card.Body>
          <b>Similarity:</b> {res.similarity} <br />
          {res.data.content}
        </Card.Body>
      </Card>)}
    </>}
  </>
}