import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Path, Defs, LinearGradient, Stop } from 'react-native-svg';
import { colors } from '../../lib/constants';

interface EquityChartProps {
  labels: string[];
  values: number[];
  height?: number;
}

function formatValue(val: number): string {
  const absVal = Math.abs(val);
  if (absVal >= 1000) {
    const formatted = (absVal / 1000).toFixed(1);
    return val < 0 ? `-$${formatted}K` : `$${formatted}K`;
  }
  return val < 0 ? `-$${absVal.toFixed(0)}` : `$${val.toFixed(0)}`;
}

export function EquityChart({ labels, values, height = 160 }: EquityChartProps) {
  if (!values?.length) {
    return (
      <View style={[styles.container, { height }]}>
        <Text style={styles.noData}>No equity data available</Text>
      </View>
    );
  }

  // Use last 30 data points for the chart
  const chartValues = values.slice(-30);
  const chartLabels = labels.slice(-30);
  const minVal = Math.min(...chartValues);
  const maxVal = Math.max(...chartValues);
  const range = maxVal - minVal || 1;

  // Calculate change based on what's displayed in the chart
  const currentValue = chartValues[chartValues.length - 1];
  const startValue = chartValues[0];
  const change = currentValue - startValue;
  const changePct = startValue !== 0 ? (change / Math.abs(startValue)) * 100 : 0;
  const isPositive = currentValue >= 0;
  const isTrendUp = change >= 0;

  // Chart dimensions
  const chartWidth = 300;
  const chartHeight = 80;
  const padding = 4;

  // Generate smooth path
  const points = chartValues.map((val, i) => {
    const x = padding + (i / (chartValues.length - 1)) * (chartWidth - padding * 2);
    const y = chartHeight - padding - ((val - minVal) / range) * (chartHeight - padding * 2);
    return { x, y };
  });

  // Create smooth bezier curve path
  let linePath = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];
    const cpx = (prev.x + curr.x) / 2;
    linePath += ` Q ${prev.x + (curr.x - prev.x) * 0.5} ${prev.y}, ${cpx} ${(prev.y + curr.y) / 2}`;
  }
  // Add final point
  const last = points[points.length - 1];
  linePath += ` L ${last.x} ${last.y}`;

  // Area path (for gradient fill)
  const areaPath = linePath + ` L ${last.x} ${chartHeight} L ${points[0].x} ${chartHeight} Z`;

  const lineColor = isPositive ? colors.signal.gain : colors.signal.loss;
  const trendColor = isTrendUp ? colors.signal.gain : colors.signal.loss;

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.label}>Equity Curve</Text>
        <View style={styles.valueRow}>
          <Text style={[styles.value, !isPositive && styles.valueLoss]}>
            {formatValue(currentValue)}
          </Text>
          <View style={[styles.changeBadge, isTrendUp ? styles.changeBadgeUp : styles.changeBadgeDown]}>
            <Text style={[styles.changeText, { color: trendColor }]}>
              {isTrendUp ? '↑' : '↓'} {Math.abs(changePct).toFixed(1)}%
            </Text>
          </View>
        </View>
      </View>

      {/* SVG Chart */}
      <View style={styles.chartContainer}>
        <Svg width="100%" height={chartHeight} viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
          <Defs>
            <LinearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
              <Stop offset="0%" stopColor={lineColor} stopOpacity="0.3" />
              <Stop offset="100%" stopColor={lineColor} stopOpacity="0.0" />
            </LinearGradient>
          </Defs>
          {/* Area fill */}
          <Path d={areaPath} fill="url(#areaGradient)" />
          {/* Line */}
          <Path
            d={linePath}
            stroke={lineColor}
            strokeWidth="2"
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </Svg>
      </View>

      {/* Footer with range */}
      <View style={styles.footer}>
        <View style={styles.dateRange}>
          <Text style={styles.dateText}>{chartLabels[0] ?? ''}</Text>
          <Text style={styles.dateText}>{chartLabels[chartLabels.length - 1] ?? ''}</Text>
        </View>
        <View style={styles.rangeRow}>
          <Text style={styles.rangeLabel}>Range:</Text>
          <Text style={styles.rangeValue}>{formatValue(minVal)} → {formatValue(maxVal)}</Text>
        </View>
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
    marginBottom: 6,
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  value: {
    fontSize: 28,
    fontWeight: '600',
    color: colors.text.primary,
    fontFamily: 'JetBrains Mono',
  },
  valueLoss: {
    color: colors.signal.loss,
  },
  changeBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
  },
  changeBadgeUp: {
    backgroundColor: colors.signal.gainDim,
  },
  changeBadgeDown: {
    backgroundColor: colors.signal.lossDim,
  },
  changeText: {
    fontSize: 13,
    fontWeight: '600',
    fontFamily: 'JetBrains Mono',
  },
  noData: {
    flex: 1,
    textAlign: 'center',
    textAlignVertical: 'center',
    color: colors.text.muted,
    fontSize: 14,
  },
  chartContainer: {
    height: 80,
    marginVertical: 8,
  },
  footer: {
    marginTop: 8,
  },
  dateRange: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  dateText: {
    fontSize: 10,
    color: colors.text.muted,
    fontFamily: 'JetBrains Mono',
  },
  rangeRow: {
    flexDirection: 'row',
    gap: 6,
  },
  rangeLabel: {
    fontSize: 10,
    color: colors.text.muted,
  },
  rangeValue: {
    fontSize: 10,
    color: colors.text.tertiary,
    fontFamily: 'JetBrains Mono',
  },
});
