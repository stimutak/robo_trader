import React, { useMemo } from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { CartesianChart, Line } from 'victory-native';
import { matchFont } from '@shopify/react-native-skia';
import { colors } from '../../lib/constants';

interface EquityChartProps {
  labels: string[];
  values: number[];
  height?: number;
}

export function EquityChart({ labels, values, height = 180 }: EquityChartProps) {
  // Use system font via matchFont (avoids arraybuffer error)
  const font = useMemo(() => {
    try {
      return matchFont({
        fontFamily: Platform.select({ ios: 'Helvetica', default: 'sans-serif' }),
        fontSize: 10,
        fontWeight: 'normal',
      });
    } catch {
      return null;
    }
  }, []);

  // Transform data for Victory Native
  const chartData = useMemo(() => {
    if (!labels?.length || !values?.length) return [];
    return labels.map((label, i) => ({
      x: i,
      equity: values[i] ?? 0,
    }));
  }, [labels, values]);

  // Calculate if positive overall
  const isPositive = useMemo(() => {
    if (!values?.length) return true;
    return values[values.length - 1] >= values[0];
  }, [values]);

  // Get current value for display
  const currentValue = values?.[values.length - 1] ?? 0;
  const startValue = values?.[0] ?? 0;
  const change = currentValue - startValue;
  const changePct = startValue !== 0 ? (change / startValue) * 100 : 0;

  if (!chartData.length) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.noData}>No equity data available</Text>
      </View>
    );
  }

  const lineColor = isPositive ? colors.signal.gain : colors.signal.loss;

  return (
    <View style={styles.container}>
      {/* Header with current value */}
      <View style={styles.header}>
        <Text style={styles.label}>Equity Curve</Text>
        <View style={styles.valueRow}>
          <Text style={styles.value}>
            ${currentValue.toLocaleString('en-US', { maximumFractionDigits: 0 })}
          </Text>
          <Text style={[styles.change, { color: lineColor }]}>
            {change >= 0 ? '+' : ''}
            {changePct.toFixed(2)}%
          </Text>
        </View>
      </View>

      {/* Chart */}
      <View style={{ height }}>
        <CartesianChart
          data={chartData}
          xKey="x"
          yKeys={['equity']}
          axisOptions={font ? {
            font,
            labelColor: colors.text.muted,
            lineColor: colors.border,
            tickCount: { x: 4, y: 4 },
            formatXLabel: (value) => {
              const idx = Math.round(value);
              if (idx >= 0 && idx < labels.length) {
                const label = labels[idx];
                if (label?.includes('-')) {
                  const parts = label.split('-');
                  return `${parts[1]}/${parts[2]}`;
                }
                return label ?? '';
              }
              return '';
            },
            formatYLabel: (value) => {
              if (value >= 1000000) return `$${(value / 1000000).toFixed(1)}M`;
              if (value >= 1000) return `$${(value / 1000).toFixed(0)}K`;
              return `$${value.toFixed(0)}`;
            },
          } : undefined}
        >
          {({ points }) => (
            <Line
              points={points.equity}
              color={lineColor}
              strokeWidth={2}
              curveType="natural"
            />
          )}
        </CartesianChart>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    marginBottom: 12,
  },
  label: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
  },
  value: {
    fontSize: 24,
    fontWeight: '600',
    color: colors.text.primary,
  },
  change: {
    fontSize: 14,
    fontWeight: '500',
  },
  noData: {
    flex: 1,
    textAlign: 'center',
    textAlignVertical: 'center',
    color: colors.text.muted,
    fontSize: 14,
  },
});
