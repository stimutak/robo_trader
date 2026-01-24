import { useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  Pressable,
  SafeAreaView,
} from 'react-native';
import { useRouter } from 'expo-router';
import Ionicons from '@expo/vector-icons/Ionicons';
import { colors } from '../../lib/constants';
import { useStatus, usePnL, usePositions, usePerformance, useMLStatus } from '../../hooks/useAPI';
import { useTradingStore } from '../../stores/trading';
import { Card } from '../../components/ui/Card';
import { StatusDot } from '../../components/ui/StatusDot';
import { Position } from '../../lib/types';

function formatCurrency(value: number, showSign = false): string {
  const absValue = Math.abs(value);
  const formatted = absValue.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  if (showSign && value !== 0) {
    return value >= 0 ? `+$${formatted}` : `-$${formatted}`;
  }
  return `$${formatted}`;
}

function formatPercent(value: number, showSign = false): string {
  const formatted = Math.abs(value).toFixed(2) + '%';
  if (showSign && value !== 0) {
    return value >= 0 ? `+${formatted}` : `-${formatted}`;
  }
  return formatted;
}

function formatDateTime(): string {
  const now = new Date();
  const date = now.toLocaleDateString('en-US', {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
  });
  const time = now.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  });
  return `${date} • ${time}`;
}

export default function HomeScreen() {
  const router = useRouter();

  const { data: statusData, refetch: refetchStatus, isLoading: statusLoading } = useStatus();
  const { data: pnlData, refetch: refetchPnL } = usePnL();
  const { data: positionsData, refetch: refetchPositions } = usePositions();
  const { data: perfData, refetch: refetchPerf } = usePerformance();
  const { data: mlData, refetch: refetchML } = useMLStatus();

  const setStatus = useTradingStore((s) => s.setStatus);
  const setPnL = useTradingStore((s) => s.setPnL);
  const setPositions = useTradingStore((s) => s.setPositions);
  const setPerformance = useTradingStore((s) => s.setPerformance);
  const setMLStatus = useTradingStore((s) => s.setMLStatus);

  // Update store when data changes
  useEffect(() => {
    if (statusData?.trading_status) {
      setStatus(statusData.trading_status);
    }
  }, [statusData, setStatus]);

  useEffect(() => {
    if (pnlData) {
      setPnL(pnlData);
    }
  }, [pnlData, setPnL]);

  useEffect(() => {
    if (positionsData?.positions) {
      setPositions(positionsData.positions);
    }
  }, [positionsData, setPositions]);

  useEffect(() => {
    if (perfData?.summary) {
      setPerformance(perfData.summary);
    }
  }, [perfData, setPerformance]);

  useEffect(() => {
    if (mlData) {
      setMLStatus(mlData);
    }
  }, [mlData, setMLStatus]);

  const handleRefresh = async () => {
    await Promise.all([
      refetchStatus(),
      refetchPnL(),
      refetchPositions(),
      refetchPerf(),
      refetchML(),
    ]);
  };

  const status = statusData?.trading_status;
  const positions = positionsData?.positions || [];
  const sortedPositions = [...positions]
    .sort((a, b) => Math.abs(b.unrealized_pnl || 0) - Math.abs(a.unrealized_pnl || 0))
    .slice(0, 5);

  const equity = pnlData?.equity || 0;
  const dailyPnL = pnlData?.daily || 0;
  const unrealized = pnlData?.unrealized || 0;
  const winRate = (perfData?.summary?.win_rate || 0) * 100;
  const sharpe = perfData?.summary?.total_sharpe || perfData?.all?.sharpe || 0;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={statusLoading}
            onRefresh={handleRefresh}
            tintColor={colors.text.secondary}
          />
        }
      >
        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Portfolio</Text>
            <Text style={styles.datetime}>{formatDateTime()}</Text>
          </View>
          <View style={styles.marketBadge}>
            <View
              style={[
                styles.marketDot,
                { backgroundColor: status?.market_open ? colors.signal.gain : colors.text.tertiary },
                status?.market_open && styles.marketDotGlow,
              ]}
            />
            <Text style={styles.marketText}>
              {status?.market_open ? 'Open' : 'Closed'}
            </Text>
          </View>
        </View>

        {/* Portfolio Hero */}
        <View style={styles.portfolioHero}>
          <Text style={styles.portLabel}>Total Equity</Text>
          <View style={styles.portValueRow}>
            <Text style={styles.portSymbol}>$</Text>
            <Text style={styles.portValue}>
              {Math.floor(equity).toLocaleString('en-US')}
            </Text>
            <Text style={styles.portDecimals}>
              .{(equity % 1).toFixed(2).substring(2)}
            </Text>
          </View>
          <View style={styles.pnlRow}>
            <View
              style={[
                styles.pnlChip,
                dailyPnL > 0 && styles.pnlChipUp,
                dailyPnL < 0 && styles.pnlChipDown,
              ]}
            >
              {dailyPnL !== 0 && (
                <Ionicons
                  name={dailyPnL > 0 ? 'arrow-up' : 'arrow-down'}
                  size={11}
                  color={dailyPnL > 0 ? colors.signal.gainBright : colors.signal.lossBright}
                />
              )}
              <Text
                style={[
                  styles.pnlValue,
                  dailyPnL > 0 && styles.pnlValueUp,
                  dailyPnL < 0 && styles.pnlValueDown,
                ]}
              >
                {formatCurrency(dailyPnL, true)}
              </Text>
            </View>
            <Text style={styles.pnlLabel}>Today</Text>
          </View>
        </View>

        {/* Stats Strip */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.statsStrip}
          contentContainerStyle={styles.statsStripContent}
        >
          <View style={styles.statChip}>
            <Text style={styles.statLabel}>Day P&L</Text>
            <Text style={[styles.statValue, dailyPnL > 0 && styles.statUp, dailyPnL < 0 && styles.statDown]}>
              {formatCurrency(dailyPnL, true)}
            </Text>
          </View>
          <View style={styles.statChip}>
            <Text style={styles.statLabel}>Unrealized</Text>
            <Text style={[styles.statValue, unrealized > 0 && styles.statUp, unrealized < 0 && styles.statDown]}>
              {formatCurrency(unrealized, true)}
            </Text>
          </View>
          <View style={styles.statChip}>
            <Text style={styles.statLabel}>Positions</Text>
            <Text style={styles.statValue}>{positions.length}</Text>
          </View>
          <View style={styles.statChip}>
            <Text style={styles.statLabel}>Win Rate</Text>
            <Text style={styles.statValue}>{formatPercent(winRate)}</Text>
          </View>
          <View style={styles.statChip}>
            <Text style={styles.statLabel}>Sharpe</Text>
            <Text style={styles.statValue}>{sharpe.toFixed(2)}</Text>
          </View>
        </ScrollView>

        {/* Positions Section */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Active Positions</Text>
          <Pressable style={styles.sectionLink}>
            <Text style={styles.sectionLinkText}>View All</Text>
          </Pressable>
        </View>

        {sortedPositions.length === 0 ? (
          <Card style={styles.emptyCard}>
            <View style={styles.emptyState}>
              <Ionicons name="analytics-outline" size={48} color={colors.text.muted} />
              <Text style={styles.emptyTitle}>No Active Positions</Text>
              <Text style={styles.emptyText}>
                Open positions will appear here with real-time P&L
              </Text>
            </View>
          </Card>
        ) : (
          <View style={styles.positionList}>
            {sortedPositions.map((position) => (
              <PositionCard
                key={position.symbol}
                position={position}
                onPress={() => router.push(`/position/${position.symbol}`)}
              />
            ))}
          </View>
        )}

        {/* System Status */}
        <Card style={styles.systemCard}>
          <View style={styles.systemRow}>
            <View style={styles.sysItem}>
              <StatusDot variant={status?.gateway_available ? 'ok' : 'off'} />
              <Text style={styles.sysLabel}>Gateway</Text>
            </View>
            <View style={styles.sysItem}>
              <StatusDot variant={status?.runner_state === 'running' ? 'ok' : 'off'} />
              <Text style={styles.sysLabel}>Runner</Text>
            </View>
            <View style={styles.sysItem}>
              <StatusDot variant={(mlData?.models_trained || 0) > 0 ? 'ok' : 'warning'} />
              <Text style={styles.sysLabel}>ML</Text>
            </View>
            <Text style={styles.sysInfo}>
              {status?.symbols_count ? `${status.symbols_count} symbols` : '--'}
            </Text>
          </View>
        </Card>
      </ScrollView>
    </SafeAreaView>
  );
}

function PositionCard({ position, onPress }: { position: Position; onPress: () => void }) {
  const pnl = position.unrealized_pnl || 0;
  const pnlPct = position.unrealized_pnl_pct || 0;
  const isWin = pnl > 0;
  const isLoss = pnl < 0;

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.positionCard,
        pressed && styles.positionCardPressed,
      ]}
    >
      <View
        style={[
          styles.positionEdge,
          isWin && styles.positionEdgeWin,
          isLoss && styles.positionEdgeLoss,
        ]}
      />
      <View style={styles.positionContent}>
        <View style={styles.posTop}>
          <View style={styles.posSymbol}>
            <View style={styles.symIcon}>
              <Text style={styles.symIconText}>
                {position.symbol.substring(0, 2)}
              </Text>
            </View>
            <View>
              <Text style={styles.symName}>{position.symbol}</Text>
              <Text style={styles.symMeta}>{position.strategy || 'Unknown'}</Text>
            </View>
          </View>
          <View style={styles.posPnl}>
            <Text
              style={[
                styles.pnlAmt,
                isWin && styles.pnlAmtUp,
                isLoss && styles.pnlAmtDown,
              ]}
            >
              {formatCurrency(pnl, true)}
            </Text>
            <Text style={styles.pnlPct}>{formatPercent(pnlPct, true)}</Text>
          </View>
        </View>
        <View style={styles.posBottom}>
          <View style={styles.posDetail}>
            <Text style={styles.posDetailLabel}>Shares</Text>
            <Text style={styles.posDetailValue}>{position.quantity}</Text>
          </View>
          <View style={styles.posDetail}>
            <Text style={styles.posDetailLabel}>Avg Cost</Text>
            <Text style={styles.posDetailValue}>
              ${(position.entry_price ?? 0).toFixed(2)}
            </Text>
          </View>
          <View style={styles.posDetail}>
            <Text style={styles.posDetailLabel}>Current</Text>
            <Text style={styles.posDetailValue}>
              ${(position.current_price ?? 0).toFixed(2)}
            </Text>
          </View>
        </View>
      </View>
    </Pressable>
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
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  greeting: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  datetime: {
    fontSize: 12,
    fontFamily: 'JetBrains Mono',
    color: colors.text.tertiary,
    marginTop: 2,
  },
  marketBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: colors.bg.surface,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
  },
  marketDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  marketDotGlow: {
    shadowColor: colors.signal.gain,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 4,
  },
  marketText: {
    fontSize: 10,
    fontWeight: '600',
    color: colors.text.secondary,
    textTransform: 'uppercase',
    letterSpacing: 0.4,
  },
  portfolioHero: {
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  portLabel: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 4,
  },
  portValueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  portSymbol: {
    fontSize: 26,
    fontFamily: 'JetBrains Mono',
    color: colors.text.secondary,
    marginRight: 2,
  },
  portValue: {
    fontSize: 42,
    fontFamily: 'JetBrains Mono',
    fontWeight: '300',
    color: colors.text.primary,
    letterSpacing: -1,
  },
  portDecimals: {
    fontSize: 26,
    fontFamily: 'JetBrains Mono',
    color: colors.text.muted,
  },
  pnlRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginTop: 10,
  },
  pnlChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: colors.bg.elevated,
    borderRadius: 6,
  },
  pnlChipUp: {
    backgroundColor: colors.signal.gainDim,
  },
  pnlChipDown: {
    backgroundColor: colors.signal.lossDim,
  },
  pnlValue: {
    fontSize: 13,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.secondary,
  },
  pnlValueUp: {
    color: colors.signal.gainBright,
  },
  pnlValueDown: {
    color: colors.signal.lossBright,
  },
  pnlLabel: {
    fontSize: 12,
    color: colors.text.muted,
  },
  statsStrip: {
    marginBottom: 16,
  },
  statsStripContent: {
    paddingHorizontal: 20,
    gap: 8,
  },
  statChip: {
    minWidth: 88,
    paddingHorizontal: 14,
    paddingVertical: 12,
    backgroundColor: colors.bg.surface,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
  },
  statLabel: {
    fontSize: 9,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 0.4,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 14,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.primary,
  },
  statUp: {
    color: colors.signal.gain,
  },
  statDown: {
    color: colors.signal.loss,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
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
  sectionLink: {
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  sectionLinkText: {
    fontSize: 12,
    fontWeight: '500',
    color: colors.signal.active,
  },
  emptyCard: {
    marginHorizontal: 20,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  emptyTitle: {
    fontSize: 15,
    fontWeight: '500',
    color: colors.text.secondary,
    marginTop: 16,
    marginBottom: 6,
  },
  emptyText: {
    fontSize: 13,
    color: colors.text.tertiary,
    textAlign: 'center',
  },
  positionList: {
    paddingHorizontal: 20,
    gap: 10,
  },
  positionCard: {
    flexDirection: 'row',
    backgroundColor: colors.bg.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  positionCardPressed: {
    backgroundColor: colors.bg.elevated,
    transform: [{ scale: 0.98 }],
  },
  positionEdge: {
    width: 3,
    backgroundColor: colors.text.tertiary,
  },
  positionEdgeWin: {
    backgroundColor: colors.signal.gain,
  },
  positionEdgeLoss: {
    backgroundColor: colors.signal.loss,
  },
  positionContent: {
    flex: 1,
    padding: 14,
  },
  posTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  posSymbol: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  symIcon: {
    width: 36,
    height: 36,
    borderRadius: 9,
    backgroundColor: colors.bg.elevated,
    alignItems: 'center',
    justifyContent: 'center',
  },
  symIconText: {
    fontSize: 10,
    fontFamily: 'JetBrains Mono',
    fontWeight: '700',
    color: colors.text.tertiary,
  },
  symName: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.text.primary,
  },
  symMeta: {
    fontSize: 11,
    color: colors.text.muted,
    marginTop: 1,
  },
  posPnl: {
    alignItems: 'flex-end',
  },
  pnlAmt: {
    fontSize: 14,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.secondary,
  },
  pnlAmtUp: {
    color: colors.signal.gain,
  },
  pnlAmtDown: {
    color: colors.signal.loss,
  },
  pnlPct: {
    fontSize: 11,
    color: colors.text.muted,
    marginTop: 1,
  },
  posBottom: {
    flexDirection: 'row',
    gap: 16,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  posDetail: {
    flex: 1,
  },
  posDetailLabel: {
    fontSize: 9,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 0.3,
    marginBottom: 2,
  },
  posDetailValue: {
    fontSize: 11,
    fontFamily: 'JetBrains Mono',
    color: colors.text.secondary,
  },
  systemCard: {
    marginHorizontal: 20,
    marginTop: 12,
    padding: 12,
  },
  systemRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  sysItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
  },
  sysLabel: {
    fontSize: 10,
    color: colors.text.secondary,
  },
  sysInfo: {
    fontSize: 10,
    fontFamily: 'JetBrains Mono',
    color: colors.text.muted,
  },
});
