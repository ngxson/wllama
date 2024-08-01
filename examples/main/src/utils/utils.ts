import { useEffect } from "react";
import { Template } from "@huggingface/jinja";
import { Message, Screen } from "./types";
import { Wllama } from '@wllama/wllama';
import { DEFAULT_CHAT_TEMPLATE } from "./config";

export const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const useDidMount = (callback: () => any) => useEffect(() => {
  callback();
}, []);

type StorageKey = 'conversations' | 'params' | 'welcome' | 'custom_models';

export const WllamaStorage = {
  save<T>(key: StorageKey, data: T) {
    localStorage.setItem(key, JSON.stringify(data));
  },
  load<T>(key: StorageKey, defaultValue: T): T {
    if (localStorage[key]) {
      return JSON.parse(localStorage[key]);
    } else {
      return defaultValue;
    }
  },
};

export const getDefaultScreen = (): Screen => {
  const welcome: boolean = WllamaStorage.load('welcome', true);
  return welcome ? Screen.GUIDE : Screen.MODEL;
};

export const formatChat = async (modelWllama: Wllama, messages: Message[]): Promise<string> => {
  const template = new Template(modelWllama.getChatTemplate() ?? DEFAULT_CHAT_TEMPLATE);
  return template.render({
    messages,
    bos_token: await modelWllama.detokenize([modelWllama.getBOS()]),
    eos_token: await modelWllama.detokenize([modelWllama.getEOS()]),
    add_generation_prompt: true,
  });
};

export const toHumanReadableSize = (bytes: number): string => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
};
