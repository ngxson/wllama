import ChatScreen from './components/ChatScreen';
import GuideScreen from './components/GuideScreen';
import ModelScreen from './components/ModelScreen';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import { MessagesProvider } from './utils/messages.context';
import { Screen } from './utils/types';
import { useWllama, WllamaProvider } from './utils/wllama.context';

function App() {
  return (
    <MessagesProvider>
      <WllamaProvider>
        <InnerApp />
      </WllamaProvider>
    </MessagesProvider>
  );
}

function InnerApp() {
  const { currScreen } = useWllama();

  return (
    <div className="flex flex-col drawer h-screen w-screen overflow-hidden">
      <Navbar />
      <div className="grow flex flex-row lg:drawer-open h-[calc(100vh-4rem)]">
        <Sidebar>
          {currScreen === Screen.MODEL && <ModelScreen />}
          {currScreen === Screen.CHAT && <ChatScreen />}
          {currScreen === Screen.GUIDE && <GuideScreen />}
        </Sidebar>
      </div>
    </div>
  );
}

export default App;
