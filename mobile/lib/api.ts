import { API_BASE } from './constants';
import {
  StatusResponse,
  PnLResponse,
  PositionsResponse,
  PerformanceResponse,
  MLStatusResponse,
  PredictionsResponse,
  TradesResponse,
  EquityCurveResponse,
} from './types';

async function fetchAPI<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) {
    throw new Error(`API Error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  getStatus: () => fetchAPI<StatusResponse>('/api/status'),
  getPnL: () => fetchAPI<PnLResponse>('/api/pnl'),
  getPositions: () => fetchAPI<PositionsResponse>('/api/positions'),
  getPerformance: () => fetchAPI<PerformanceResponse>('/api/performance'),
  getMLStatus: () => fetchAPI<MLStatusResponse>('/api/ml/status'),
  getPredictions: () => fetchAPI<PredictionsResponse>('/api/ml/predictions'),
  getTrades: () => fetchAPI<TradesResponse>('/api/trades'),
  getEquityCurve: () => fetchAPI<EquityCurveResponse>('/api/equity-curve'),
};

export default api;
