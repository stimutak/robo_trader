// Log entry from WebSocket or API
export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  source: string;
  message: string;
  context?: Record<string, any>;
}

// Position data
export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  market_price: number;
  pnl: number;
  pnl_pct: number;
  sector?: string;
}

// Trade data
export interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  pnl?: number;
  timestamp: string;
}

// ML Prediction
export interface Prediction {
  symbol: string;
  signal: -1 | 0 | 1; // SELL, HOLD, BUY
  confidence: number;
  model?: string;
}

// API Response Types
export interface StatusResponse {
  trading_status: {
    is_trading: boolean;
    api_connected: boolean;
    gateway_available: boolean;
    market_open: boolean;
    runner_state: string;
    symbols_count: number;
  };
}

export interface PnLResponse {
  daily: number;
  total: number;
  unrealized: number;
  equity: number;
}

export interface PositionsResponse {
  positions: Position[];
}

export interface PerformanceResponse {
  summary: {
    win_rate: number;
    sharpe_ratio: number;
    max_drawdown: number;
    profit_factor: number;
    avg_win: number;
    avg_loss: number;
  };
}

export interface MLStatusResponse {
  models_trained: number;
  feature_count: number;
  last_training: string | null;
  accuracy: number;
}

export interface PredictionsResponse {
  predictions: Prediction[];
}

export interface TradesResponse {
  trades: Trade[];
}

export interface EquityCurveResponse {
  labels: string[];
  values: number[];
  pnl_by_trade: number[];
}

// WebSocket message types
export type WSMessageType =
  | 'log'
  | 'status_update'
  | 'position_update'
  | 'trade_executed'
  | 'pnl_update';

export interface WSMessage {
  type: WSMessageType;
  data?: any;
  timestamp?: string;
  level?: string;
  source?: string;
  message?: string;
  context?: Record<string, any>;
}
