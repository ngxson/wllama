import { useState } from 'react';
import { Container } from 'react-bootstrap';
import TopBar from './TopBar';
import { AppContextProvider, useAppContext } from './utils';
import TabQuery from './TabQuery';
import TabDataset from './TabDataset';

type ViewMode = 'query' | 'dataset';

function App() {
  const [view, setView] = useState<ViewMode>('query');

  return <>
    <TopBar onChangeTab={setView} />
    <Container className='mb-4'>
      <br/><br/>
      <h2>Retrieval demo - Wllama</h2>
      <br/>
      <AppContextProvider>
        <AppInner view={view} />
      </AppContextProvider>
    </Container>
  </>;
}

function AppInner({ view }: { view: ViewMode }) {
  const { isModelLoaded } = useAppContext();
  if (!isModelLoaded) {
    return <>
      Loading model...
    </>;
  } else {
    return <>
      {view === 'query' && <TabQuery />}
      {view === 'dataset' && <TabDataset />}
    </>;
  }
}

export default App;
