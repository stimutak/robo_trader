import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  SafeAreaView,
} from 'react-native';
import { colors } from '../../lib/constants';
import { usePerformance, useEquityCurve } from '../../hooks/useAPI';
import { Card } from '../../components/ui/Card';

function formatCurrency(value: number): string {
  const absValue = Math.abs(value);
  const formatted = absValue.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });
  return value >= 0 ? `$${formatted}` : `-$${formatted}`;
}

function formatPercent(value: number): string {
  return (value * 100).toFixed(1) + '%';
}

export default function AnalyticsScreen() {
  const { data: perfData, refetch: refetchPerf, isLoading } = usePerformance();
  const { data: equityData, refetch: refetchEquity } = useEquityCurve();

  const handleRefresh = async () => {
    await Promise.all([refetchPerf(), refetchEquity()]);
  };

  const summary = perfData?.summary ?? {
    sharpe_ratio: 0,
    max_drawdown: 0,
    win_rate: 0,
    profit_factor: 0,
    avg_win: 0,
    avg_loss: 0,
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={isLoading}
            onRefresh={handleRefresh}
            tintColor={colors.text.secondary}
          />
        }
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Analytics</Text>
        </View>

        {/* Equity Curve Placeholder */}
        <Card style={styles.chartCard}>
          <View style={styles.chartPlaceholder}>
            <Text style={styles.chartPlaceholderText}>Equity Curve Chart</Text>
            <Text style={styles.chartPlaceholderSubtext}>
              Victory Native chart coming soon
            </Text>
          </View>
        </Card>

        {/* Section Header */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Performance Metrics</Text>
        </View>

        {/* Metrics Grid */}
        <View style={styles.metricsGrid}>
          <MetricCard
            label="Sharpe Ratio"
            value={(summary.sharpe_ratio || 0).toFixed(2)}
            sub="Risk-adjusted return"
          />
          <MetricCard
            label="Max Drawdown"
            value={formatPercent(summary.max_drawdown || 0)}
            sub="Peak to trough"
            variant="loss"
          />
          <MetricCard
            label="Win Rate"
            value={formatPercent(summary.win_rate || 0)}
            sub="Winning trades"
          />
          <MetricCard
            label="Profit Factor"
            value={(summary.profit_factor || 0).toFixed(2)}
            sub="Gross P / Gross L"
          />
          <MetricCard
            label="Avg Win"
            value={formatCurrency(summary.avg_win || 0)}
            sub="Per winning trade"
            variant="gain"
          />
          <MetricCard
            label="Avg Loss"
            value={formatCurrency(summary.avg_loss || 0)}
            sub="Per losing trade"
            variant="loss"
          />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function MetricCard({
  label,
  value,
  sub,
  variant,
}: {
  label: string;
  value: string;
  sub: string;
  variant?: 'gain' | 'loss';
}) {
  return (
    <Card style={styles.metricCard}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text
        style={[
          styles.metricValue,
          variant === 'gain' && styles.metricValueGain,
          variant === 'loss' && styles.metricValueLoss,
        ]}
      >
        {value}
      </Text>
      <Text style={styles.metricSub}>{sub}</Text>
    </Card>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.deep,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 20,
  },
  header: {
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  title: {
    fontSize: 22,
    fontWeight: '600',
    color: colors.text.primary,
  },
  chartCard: {
    marginHorizontal: 20,
    marginBottom: 16,
    height: 200,
  },
  chartPlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  chartPlaceholderText: {
    fontSize: 14,
    color: colors.text.tertiary,
  },
  chartPlaceholderSubtext: {
    fontSize: 12,
    color: colors.text.muted,
    marginTop: 4,
  },
  sectionHeader: {
    paddingHorizontal: 20,
    paddingTop: 14,
    paddingBottom: 8,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 15,
    gap: 10,
  },
  metricCard: {
    width: '47%',
    padding: 14,
  },
  metricLabel: {
    fontSize: 10,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 6,
  },
  metricValue: {
    fontSize: 20,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.primary,
    marginBottom: 2,
  },
  metricValueGain: {
    color: colors.signal.gain,
  },
  metricValueLoss: {
    color: colors.signal.loss,
  },
  metricSub: {
    fontSize: 11,
    color: colors.text.muted,
  },
});
