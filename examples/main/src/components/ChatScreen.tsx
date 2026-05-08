import { useRef, useState } from 'react';
import { useMessages } from '../utils/messages.context';
import { useWllama } from '../utils/wllama.context';
import { MediaData, Message, Screen } from '../utils/types';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faStop } from '@fortawesome/free-solid-svg-icons';
import ScreenWrapper from './ScreenWrapper';
import { useIntervalWhen } from '../utils/use-interval-when';
import { MarkdownMessage } from './MarkdownMessage';

export default function ChatScreen() {
  const [input, setInput] = useState('');
  const [pendingMedia, setPendingMedia] = useState<MediaData | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);
  const {
    currentConvId,
    isGenerating,
    createCompletion,
    navigateTo,
    loadedModel,
    stopCompletion,
    currRuntimeInfo,
  } = useWllama();
  const {
    getConversationById,
    addMessageToConversation,
    editMessageInConversation,
    newConversation,
  } = useMessages();

  useIntervalWhen(chatScrollToBottom, 500, isGenerating, true);

  const currConv = getConversationById(currentConvId);
  const supportsMedia =
    currRuntimeInfo?.supportsImage || currRuntimeInfo?.supportsAudio;

  const onPickFile =
    (type: 'image' | 'audio') => (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        setPendingMedia({
          type,
          data: ev.target!.result as ArrayBuffer,
          dataUrl: URL.createObjectURL(file),
        });
      };
      reader.readAsArrayBuffer(file);
      e.target.value = '';
    };

  const onSubmit = async () => {
    if (isGenerating) return;

    const currHistory = currConv?.messages ?? [];
    const userInput = input;
    const media = pendingMedia;
    setInput('');
    setPendingMedia(null);
    const userMsg: Message = {
      id: Date.now(),
      content: userInput,
      role: 'user',
      mediaData: media ?? undefined,
    };
    const assistantMsg: Message = {
      id: Date.now() + 1,
      content: '',
      role: 'assistant',
    };

    let convId = currConv?.id;
    if (!convId) {
      const newConv = newConversation(userMsg);
      convId = newConv.id;
      navigateTo(Screen.CHAT, convId);
      addMessageToConversation(convId, assistantMsg);
    } else {
      addMessageToConversation(convId, userMsg);
      addMessageToConversation(convId, assistantMsg);
    }

    await createCompletion([...currHistory, userMsg], (newContent) => {
      editMessageInConversation(convId, assistantMsg.id, newContent);
    });
  };

  return (
    <ScreenWrapper fitScreen>
      <div className="chat-messages grow overflow-auto" id="chat-history">
        <div className="h-10" />

        {currConv ? (
          <>
            {currConv.messages.map((msg) =>
              msg.role === 'user' ? (
                <div className="chat chat-end" key={msg.id}>
                  <div className="chat-bubble">
                    {msg.mediaData?.type === 'image' && (
                      <img
                        src={msg.mediaData.dataUrl}
                        className="max-w-48 rounded mb-1"
                      />
                    )}
                    {msg.mediaData?.type === 'audio' && (
                      <audio
                        controls
                        src={msg.mediaData.dataUrl}
                        className="mb-1"
                      />
                    )}
                    {msg.content.length > 0 && (
                      <MarkdownMessage content={msg.content} />
                    )}
                  </div>
                </div>
              ) : (
                <div className="chat chat-start" key={msg.id}>
                  <div className="chat-bubble bg-base-100 text-base-content">
                    {msg.content.length === 0 && isGenerating && (
                      <span className="loading loading-dots"></span>
                    )}
                    {msg.content.length > 0 && (
                      <MarkdownMessage content={msg.content} />
                    )}
                  </div>
                </div>
              )
            )}
          </>
        ) : (
          <div className="pt-24 text-center text-xl">Ask me something 👋</div>
        )}
      </div>
      <div className="flex flex-col input-message py-4">
        {isGenerating && (
          <div className="text-center">
            <button
              className="btn btn-outline btn-sm mb-4"
              onClick={stopCompletion}
            >
              <FontAwesomeIcon icon={faStop} />
              Stop generation
            </button>
          </div>
        )}

        {loadedModel && (
          <>
            {pendingMedia && (
              <div className="flex items-center gap-2 mb-2">
                {pendingMedia.type === 'image' ? (
                  <img
                    src={pendingMedia.dataUrl}
                    className="h-16 w-16 object-cover rounded"
                  />
                ) : (
                  <audio controls src={pendingMedia.dataUrl} />
                )}
                <button
                  className="btn btn-xs btn-circle btn-outline"
                  onClick={() => setPendingMedia(null)}
                >
                  ✕
                </button>
              </div>
            )}
            <div className="flex gap-2">
              {supportsMedia && (
                <>
                  <input
                    ref={imageInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={onPickFile('image')}
                  />
                  <input
                    ref={audioInputRef}
                    type="file"
                    accept="audio/*"
                    className="hidden"
                    onChange={onPickFile('audio')}
                  />
                  <div className="dropdown dropdown-top">
                    <button
                      tabIndex={0}
                      className="btn btn-sm btn-ghost h-full border border-base-content/20"
                      disabled={isGenerating}
                    >
                      +
                    </button>
                    <ul className="dropdown-content menu bg-base-100 rounded-box z-[1] w-36 p-2 shadow mb-1">
                      {currRuntimeInfo?.supportsImage && (
                        <li>
                          <a
                            onMouseDown={(e) => {
                              e.preventDefault();
                              imageInputRef.current?.click();
                            }}
                          >
                            Image
                          </a>
                        </li>
                      )}
                      {currRuntimeInfo?.supportsAudio && (
                        <li>
                          <a
                            onMouseDown={(e) => {
                              e.preventDefault();
                              audioInputRef.current?.click();
                            }}
                          >
                            Audio
                          </a>
                        </li>
                      )}
                    </ul>
                  </div>
                </>
              )}
              <textarea
                className="textarea textarea-bordered grow"
                placeholder="Your message..."
                disabled={isGenerating}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.keyCode == 13 && e.shiftKey == false) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
              />
            </div>
          </>
        )}

        {!loadedModel && <WarnNoModel />}

        <small className="text-center mx-auto opacity-70 pt-2">
          wllama may generate inaccurate information. Use with your own risk.
        </small>
      </div>
    </ScreenWrapper>
  );
}

function WarnNoModel() {
  const { navigateTo } = useWllama();

  return (
    <div role="alert" className="alert">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className="h-6 w-6 shrink-0 stroke-current"
        fill="none"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
      <span>Model is not loaded</span>
      <div>
        <button
          className="btn btn-sm btn-primary"
          onClick={() => navigateTo(Screen.MODEL)}
        >
          Select model
        </button>
      </div>
    </div>
  );
}

const chatScrollToBottom = () => {
  const elem = document.getElementById('chat-history');
  elem?.scrollTo({
    top: elem.scrollHeight,
    behavior: 'smooth',
  });
};
