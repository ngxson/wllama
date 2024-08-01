export enum Screen {
  GUIDE,
  CHAT,
  MODEL,
}

export interface Model {
  url: string;
  size: string;
  formatChat(messages: Message[]): string;
}

export enum ModelState {
  NOT_DOWNLOADED,
  DOWNLOADING,
  READY,
  LOADING,
  LOADED,
}

export interface ManageModel extends Model {
  state: ModelState;
  downloadPercent: number; // from 0.0 to 1.0
}

export interface InferenceParams {
  nThreads: number;
  nContext: number;
  temperature: number;
  nPredict: number;
}

export interface Message {
  id: number;
  content: string;
  role: 'user' | 'assistant';
}

export interface Conversation {
  id: number;
  messages: Message[];
}
