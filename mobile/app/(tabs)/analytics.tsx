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
import { EquityChart } from '../../components/charts';
import { Skeleton, SkeletonChart } from '../../components/ui/Skeleton';
import { ErrorState } from '../../components/ui/ErrorState';

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
  const { data: perfData, refetch: refetchPerf, isLoading, error: perfError } = usePerformance();
  const { data: equityData, refetch: refetchEquity, error: equityError } = useEquityCurve();

  const handleRefresh = async () => {
    await Promise.all([refetchPerf(), refetchEquity()]);
  };

  const hasError = perfError || equityError;

  // Map API fields to display values
  const sharpeRatio = perfData?.summary?.total_sharpe ?? perfData?.all?.sharpe ?? 0;
  const maxDrawdown = perfData?.summary?.total_drawdown ?? perfData?.all?.max_drawdown ?? 0;
  const winRate = perfData?.summary?.win_rate ?? 0;
  const totalPnl = perfData?.summary?.total_pnl ?? perfData?.all?.pnl ?? 0;
  const totalTrades = perfData?.summary?.total_trades ?? perfData?.all?.trades ?? 0;
  const winningTrades = perfData?.summary?.winning_trades ?? 0;
  const losingTrades = perfData?.summary?.losing_trades ?? 0;

  // Error state
  if (hasError && !perfData && !equityData) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Analytics</Text>
        </View>
        <ErrorState
          title="Unable to Load"
          message="Could not fetch performance data. Check your connection."
          onRetry={handleRefresh}
        />
      </SafeAreaView>
    );
  }

  // Loading skeleton
  if (isLoading && !perfData && !equityData) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Analytics</Text>
        </View>
        <View style={{ paddingHorizontal: 20 }}>
          <SkeletonChart />
        </View>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Performance Metrics</Text>
        </View>
        <View style={styles.metricsGrid}>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Card key={i} style={styles.metricCard}>
              <Skeleton width={60} height={10} />
              <Skeleton width={50} height={20} style={{ marginTop: 8 }} />
              <Skeleton width={70} height={10} style={{ marginTop: 6 }} />
            </Card>
          ))}
        </View>
      </SafeAreaView>
    );
  }

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

        {/* Equity Curve Chart */}
        <Card style={styles.chartCard}>
          <EquityChart
            labels={equityData?.labels ?? []}
            values={equityData?.values ?? []}
            height={160}
          />
        </Card>

        {/* Section Header */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Performance Metrics</Text>
        </View>

        {/* Metrics Grid */}
        <View style={styles.metricsGrid}>
          <MetricCard
            label="Sharpe Ratio"
            value={sharpeRatio.toFixed(2)}
            sub="Risk-adjusted return"
          />
          <MetricCard
            label="Max Drawdown"
            value={formatPercent(maxDrawdown)}
            sub="Peak to trough"
            variant="loss"
          />
          <MetricCard
            label="Win Rate"
            value={formatPercent(winRate)}
            sub="Winning trades"
          />
          <MetricCard
            label="Total P&L"
            value={formatCurrency(totalPnl)}
            sub={`${totalTrades} trades`}
            variant={totalPnl >= 0 ? 'gain' : 'loss'}
          />
          <MetricCard
            label="Wins"
            value={winningTrades.toString()}
            sub="Winning trades"
            variant="gain"
          />
          <MetricCard
            label="Losses"
            value={losingTrades.toString()}
            sub="Losing trades"
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
    padding: 16,
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
