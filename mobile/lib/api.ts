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
  const url = `${API_BASE}${endpoint}`;
  console.log(`[API] Fetching: ${url}`);
  try {
    const res = await fetch(url);
    console.log(`[API] Response status: ${res.status} for ${endpoint}`);
    if (!res.ok) {
      throw new Error(`API Error: ${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    console.log(`[API] Data received for ${endpoint}:`, JSON.stringify(data).slice(0, 200));
    return data;
  } catch (error) {
    console.error(`[API] Error fetching ${endpoint}:`, error);
    throw error;
  }
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
