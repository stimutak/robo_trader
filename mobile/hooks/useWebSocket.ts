import { useEffect, useRef, useCallback } from 'react';
import ReconnectingWebSocket from 'reconnecting-websocket';
import { useLogsStore } from '../stores/logs';
import { WS_URL } from '../lib/constants';
import { LogEntry, WSMessage } from '../lib/types';

export function useWebSocket() {
  const wsRef = useRef<ReconnectingWebSocket | null>(null);
  const addLog = useLogsStore((s) => s.addLog);
  const setWsConnected = useLogsStore((s) => s.setWsConnected);

  useEffect(() => {
    const ws = new ReconnectingWebSocket(WS_URL, [], {
      maxReconnectionDelay: 10000,
      minReconnectionDelay: 1000,
      reconnectionDelayGrowFactor: 1.3,
      maxRetries: Infinity,
    });

    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      // Subscribe to logs channel
      ws.send(JSON.stringify({ type: 'subscribe', channel: 'logs' }));
    };

    ws.onclose = () => {
      setWsConnected(false);
    };

    ws.onerror = () => {
      setWsConnected(false);
    };

    ws.onmessage = (event) => {
      try {
        const data: WSMessage = JSON.parse(event.data);

        if (data.type === 'log') {
          const logEntry: LogEntry = {
            id: data.timestamp || Date.now().toString(),
            timestamp: data.timestamp || new Date().toISOString(),
            level: (data.level as LogEntry['level']) || 'INFO',
            source: data.source || 'ws',
            message: data.message || '',
            context: data.context,
          };
          addLog(logEntry);
        }
        // Handle other message types as needed
      } catch {
        // Plain text message - treat as log
        addLog({
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          level: 'INFO',
          source: 'ws',
          message: event.data,
        });
      }
    };

    return () => {
      ws.close();
    };
  }, [addLog, setWsConnected]);

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send };
}
