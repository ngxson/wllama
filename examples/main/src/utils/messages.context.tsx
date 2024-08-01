import { createContext, useContext, useMemo, useState } from 'react';
import { Conversation, Message } from './types';
import { WllamaStorage } from './utils';

interface MessagesContextValue {
  conversations: Conversation[];
  newConversation: (message: Message) => Conversation;
  deleteConversation: (id: number) => void;
  addMessageToConversation: (id: number, message: Message) => void;
  getConversationById: (id: number) => Conversation | undefined;
  editMessageInConversation: (
    conversationId: number,
    messageId: number,
    content: string
  ) => void;
}

const MessagesContext = createContext<MessagesContextValue>({} as any);

type ConvMap = { [id: number]: Conversation };

export const MessagesProvider = ({ children }: any) => {
  const [conversations, _setConversations] = useState<ConvMap>(
    WllamaStorage.load('conversations', {})
  );
  const sortedConversations = useMemo(() => {
    return Object.values(conversations).sort((a, b) => {
      const lastMessageA = a.messages[a.messages.length - 1];
      const lastMessageB = b.messages[b.messages.length - 1];
      return lastMessageB.id - lastMessageA.id;
    });
  }, [conversations]);

  // proxy function for saving to localStorage
  const setConversations = (fn: (prev: ConvMap) => ConvMap) => {
    _setConversations((prev) => {
      const next = fn(prev);
      WllamaStorage.save('conversations', next);
      return next;
    });
  };

  const newConversation = (message: Message) => {
    const conv: Conversation = {
      id: Date.now(),
      messages: [message],
    };
    setConversations((prevConversations) => ({
      ...prevConversations,
      [conv.id]: conv,
    }));
    return conv;
  };

  const deleteConversation = (conversationId: number) => {
    setConversations((prevConversations) => {
      const newConversations = { ...prevConversations };
      delete newConversations[conversationId];
      return newConversations;
    });
  };

  const addMessageToConversation = (
    conversationId: number,
    message: Message
  ) => {
    setConversations((prevConversations) => {
      if (prevConversations[conversationId]) {
        const newConversations = { ...prevConversations };
        const conv = newConversations[conversationId];
        newConversations[conversationId].messages = [...conv.messages, message];
        return newConversations;
      } else {
        return prevConversations;
      }
    });
  };

  const editMessageInConversation = (
    conversationId: number,
    messageId: number,
    content: string
  ) => {
    setConversations((prevConversations) => {
      if (prevConversations[conversationId]) {
        const newConversations = { ...prevConversations };
        const conv = newConversations[conversationId];
        const updatedMessages = conv.messages.map((message) => {
          if (message.id === messageId) {
            return { ...message, content };
          }
          return message;
        });
        newConversations[conversationId].messages = updatedMessages;
        return newConversations;
      } else {
        return prevConversations;
      }
    });
  };

  const getConversationById = (id: number) => conversations[id];

  return (
    <MessagesContext.Provider
      value={{
        conversations: sortedConversations,
        newConversation,
        deleteConversation,
        addMessageToConversation,
        getConversationById,
        editMessageInConversation,
      }}
    >
      {children}
    </MessagesContext.Provider>
  );
};

export const useMessages = () => useContext(MessagesContext);
