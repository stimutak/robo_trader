import { useQuery } from '@tanstack/react-query';
import { api } from '../lib/api';
import { POLL_INTERVAL } from '../lib/constants';

export function useStatus() {
  return useQuery({
    queryKey: ['status'],
    queryFn: api.getStatus,
    refetchInterval: POLL_INTERVAL,
  });
}

export function usePnL() {
  return useQuery({
    queryKey: ['pnl'],
    queryFn: api.getPnL,
    refetchInterval: POLL_INTERVAL,
  });
}

export function usePositions() {
  return useQuery({
    queryKey: ['positions'],
    queryFn: api.getPositions,
    refetchInterval: POLL_INTERVAL,
  });
}

export function usePerformance() {
  return useQuery({
    queryKey: ['performance'],
    queryFn: api.getPerformance,
    refetchInterval: POLL_INTERVAL,
  });
}

export function useMLStatus() {
  return useQuery({
    queryKey: ['ml-status'],
    queryFn: api.getMLStatus,
    refetchInterval: POLL_INTERVAL,
  });
}

export function usePredictions() {
  return useQuery({
    queryKey: ['predictions'],
    queryFn: api.getPredictions,
    refetchInterval: POLL_INTERVAL,
  });
}

export function useTrades() {
  return useQuery({
    queryKey: ['trades'],
    queryFn: api.getTrades,
    refetchInterval: POLL_INTERVAL,
  });
}

export function useEquityCurve() {
  return useQuery({
    queryKey: ['equity-curve'],
    queryFn: api.getEquityCurve,
    refetchInterval: POLL_INTERVAL * 2, // Less frequent
  });
}
