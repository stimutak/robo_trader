/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,jsx,ts,tsx}", "./components/**/*.{js,jsx,ts,tsx}"],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        bg: {
          deep: '#08080a',
          surface: '#101012',
          elevated: '#161619',
          hover: '#1c1c20',
        },
        signal: {
          gain: '#10b981',
          'gain-dim': 'rgba(16, 185, 129, 0.12)',
          loss: '#ef4444',
          'loss-dim': 'rgba(239, 68, 68, 0.12)',
          warning: '#f59e0b',
          'warn-dim': 'rgba(245, 158, 11, 0.12)',
          active: '#3b82f6',
          'active-dim': 'rgba(59, 130, 246, 0.12)',
          purple: '#8b5cf6',
          'purple-dim': 'rgba(139, 92, 246, 0.12)',
        },
        text: {
          primary: '#f9fafb',
          secondary: '#9ca3af',
          tertiary: '#6b7280',
          muted: '#4b5563',
        },
      },
      fontFamily: {
        sans: ['Outfit'],
        mono: ['JetBrains Mono'],
      },
    },
  },
  plugins: [],
};
