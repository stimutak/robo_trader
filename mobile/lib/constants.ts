// API Configuration
// Use Mac's IP for iPhone access (localhost won't work on device)
const DEV_HOST = '192.168.1.166';
export const API_BASE = __DEV__ ? `http://${DEV_HOST}:5555` : 'http://localhost:5555';
export const WS_URL = __DEV__ ? `ws://${DEV_HOST}:8765` : 'ws://localhost:8765';
export const POLL_INTERVAL = 5000;
export const LOG_BUFFER_SIZE = 500;

// Color palette - dark theme only
export const colors = {
  bg: {
    deep: '#08080a',
    surface: '#101012',
    elevated: '#161619',
    hover: '#1c1c20',
  },
  signal: {
    gain: '#10b981',
    gainBright: '#34d399',
    gainDim: 'rgba(16, 185, 129, 0.12)',
    loss: '#ef4444',
    lossBright: '#f87171',
    lossDim: 'rgba(239, 68, 68, 0.12)',
    warning: '#f59e0b',
    warnDim: 'rgba(245, 158, 11, 0.12)',
    active: '#3b82f6',
    activeDim: 'rgba(59, 130, 246, 0.12)',
    purple: '#8b5cf6',
    purpleDim: 'rgba(139, 92, 246, 0.12)',
  },
  text: {
    primary: '#f9fafb',
    secondary: '#9ca3af',
    tertiary: '#6b7280',
    muted: '#4b5563',
  },
  log: {
    debug: '#6b7280',
    info: '#3b82f6',
    warning: '#f59e0b',
    error: '#ef4444',
  },
  border: 'rgba(255, 255, 255, 0.06)',
};

// Typography
export const fonts = {
  sans: 'Outfit',
  mono: 'JetBrains Mono',
};

export const fontSizes = {
  xs: 10,
  sm: 12,
  base: 14,
  lg: 16,
  xl: 20,
  '2xl': 24,
  '3xl': 32,
  '4xl': 44,
};

// Spacing
export const spacing = {
  cardPadding: 14,
  cardRadius: 14,
  listGap: 10,
  sectionGap: 16,
};

// Log level colors
export const logLevelColors: Record<string, string> = {
  DEBUG: colors.log.debug,
  INFO: colors.log.info,
  WARNING: colors.log.warning,
  ERROR: colors.log.error,
};

// Log level background colors
export const logLevelBgColors: Record<string, string> = {
  DEBUG: 'rgba(107, 114, 128, 0.2)',
  INFO: colors.signal.activeDim,
  WARNING: colors.signal.warnDim,
  ERROR: colors.signal.lossDim,
};
