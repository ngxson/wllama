export enum Screen {
  GUIDE,
  CHAT,
  MODEL,
  LOG,
}

export enum ModelState {
  NOT_DOWNLOADED,
  DOWNLOADING,
  READY,
  LOADING,
  LOADED,
}

export interface RuntimeInfo {
  isMultithread: boolean;
  hasChatTemplate: boolean;
}

export interface InferenceParams {
  nThreads: number;
  nContext: number;
  nBatch: number;
  temperature: number;
  nPredict: number;
}

export interface Message {
  id: number;
  content: string;
  role: 'system' | 'user' | 'assistant';
}

export interface Conversation {
  id: number;
  messages: Message[];
}
