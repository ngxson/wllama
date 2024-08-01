import { useEffect } from "react";

export const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const useDidMount = (callback: () => any) => useEffect(() => {
  callback();
}, []);

type StorageKey = 'conversations' | 'params';

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
