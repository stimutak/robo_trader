import { create } from 'zustand';
import { Position, Trade, Prediction } from '../lib/types';

interface TradingState {
  // System status
  isTrading: boolean;
  apiConnected: boolean;
  gatewayAvailable: boolean;
  marketOpen: boolean;
  runnerState: string;
  symbolsCount: number;

  // Portfolio
  equity: number;
  dailyPnL: number;
  unrealizedPnL: number;
  totalPnL: number;

  // Positions
  positions: Position[];

  // Performance
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;

  // ML
  modelsCount: number;
  featureCount: number;
  mlAccuracy: number;
  lastTraining: string | null;
  predictions: Prediction[];

  // Trades
  trades: Trade[];

  // Actions
  setStatus: (status: {
    is_trading: boolean;
    api_connected: boolean;
    gateway_available: boolean;
    market_open: boolean;
    runner_state: string;
    symbols_count: number;
  }) => void;
  setPnL: (pnl: {
    equity: number;
    daily: number;
    unrealized: number;
    total: number;
  }) => void;
  setPositions: (positions: Position[]) => void;
  setPerformance: (perf: {
    win_rate: number;
    sharpe_ratio: number;
    max_drawdown: number;
    profit_factor: number;
    avg_win: number;
    avg_loss: number;
  }) => void;
  setMLStatus: (ml: {
    models_trained: number;
    feature_count: number;
    accuracy: number;
    last_training: string | null;
  }) => void;
  setPredictions: (predictions: Prediction[]) => void;
  setTrades: (trades: Trade[]) => void;
}

export const useTradingStore = create<TradingState>((set) => ({
  // Initial state
  isTrading: false,
  apiConnected: false,
  gatewayAvailable: false,
  marketOpen: false,
  runnerState: 'stopped',
  symbolsCount: 0,

  equity: 0,
  dailyPnL: 0,
  unrealizedPnL: 0,
  totalPnL: 0,

  positions: [],

  winRate: 0,
  sharpeRatio: 0,
  maxDrawdown: 0,
  profitFactor: 0,
  avgWin: 0,
  avgLoss: 0,

  modelsCount: 0,
  featureCount: 0,
  mlAccuracy: 0,
  lastTraining: null,
  predictions: [],

  trades: [],

  // Actions
  setStatus: (status) =>
    set({
      isTrading: status.is_trading,
      apiConnected: status.api_connected,
      gatewayAvailable: status.gateway_available,
      marketOpen: status.market_open,
      runnerState: status.runner_state,
      symbolsCount: status.symbols_count,
    }),

  setPnL: (pnl) =>
    set({
      equity: pnl.equity,
      dailyPnL: pnl.daily,
      unrealizedPnL: pnl.unrealized,
      totalPnL: pnl.total,
    }),

  setPositions: (positions) => set({ positions }),

  setPerformance: (perf) =>
    set({
      winRate: perf.win_rate,
      sharpeRatio: perf.sharpe_ratio,
      maxDrawdown: perf.max_drawdown,
      profitFactor: perf.profit_factor,
      avgWin: perf.avg_win,
      avgLoss: perf.avg_loss,
    }),

  setMLStatus: (ml) =>
    set({
      modelsCount: ml.models_trained,
      featureCount: ml.feature_count,
      mlAccuracy: ml.accuracy,
      lastTraining: ml.last_training,
    }),

  setPredictions: (predictions) => set({ predictions }),

  setTrades: (trades) => set({ trades }),
}));
