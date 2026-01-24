// Log entry from WebSocket or API
export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  source: string;
  message: string;
  context?: Record<string, any>;
}

// Position data (matches API response)
export interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  market_value: number;
  ml_signal?: string;
  side?: string;
  strategy?: string;
  entry_time?: string;
}

// Trade data (matches API response)
export interface Trade {
  id: number;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  notional?: number;
  cash_impact?: number;
  commission?: number;
  slippage?: number;
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
    total_sharpe: number;
    total_drawdown: number;
    total_pnl: number;
    total_return: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
  };
  all?: {
    sharpe: number;
    max_drawdown: number;
    pnl: number;
    return_pct: number;
    trades: number;
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
