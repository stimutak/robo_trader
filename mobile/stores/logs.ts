import { create } from 'zustand';
import { LogEntry } from '../lib/types';
import { LOG_BUFFER_SIZE } from '../lib/constants';

export type LogLevel = 'all' | 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';

interface LogsState {
  logs: LogEntry[];
  filter: LogLevel;
  search: string;
  autoScroll: boolean;
  wsConnected: boolean;

  // Actions
  addLog: (log: LogEntry) => void;
  addLogs: (logs: LogEntry[]) => void;
  setFilter: (filter: LogLevel) => void;
  setSearch: (search: string) => void;
  toggleAutoScroll: () => void;
  setAutoScroll: (value: boolean) => void;
  setWsConnected: (connected: boolean) => void;
  clear: () => void;

  // Selectors
  getFilteredLogs: () => LogEntry[];
}

export const useLogsStore = create<LogsState>((set, get) => ({
  logs: [],
  filter: 'all',
  search: '',
  autoScroll: true,
  wsConnected: false,

  addLog: (log) =>
    set((state) => ({
      logs: [...state.logs.slice(-(LOG_BUFFER_SIZE - 1)), log],
    })),

  addLogs: (newLogs) =>
    set((state) => {
      const combined = [...state.logs, ...newLogs];
      return {
        logs: combined.slice(-LOG_BUFFER_SIZE),
      };
    }),

  setFilter: (filter) => set({ filter }),

  setSearch: (search) => set({ search }),

  toggleAutoScroll: () => set((state) => ({ autoScroll: !state.autoScroll })),

  setAutoScroll: (value) => set({ autoScroll: value }),

  setWsConnected: (connected) => set({ wsConnected: connected }),

  clear: () => set({ logs: [] }),

  getFilteredLogs: () => {
    const { logs, filter, search } = get();
    return logs.filter((log) => {
      // Filter by level
      if (filter !== 'all' && log.level !== filter) {
        return false;
      }
      // Filter by search
      if (
        search &&
        !log.message.toLowerCase().includes(search.toLowerCase())
      ) {
        return false;
      }
      return true;
    });
  },
}));
