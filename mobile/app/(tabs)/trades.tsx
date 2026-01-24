import {
  View,
  Text,
  StyleSheet,
  FlatList,
  RefreshControl,
  SafeAreaView,
  Pressable,
  ScrollView,
} from 'react-native';
import { useState, useCallback } from 'react';
import Ionicons from '@expo/vector-icons/Ionicons';
import * as Haptics from 'expo-haptics';
import { colors } from '../../lib/constants';
import { useTrades } from '../../hooks/useAPI';
import { Card } from '../../components/ui/Card';
import { Trade } from '../../lib/types';

type FilterType = 'all' | 'buys' | 'sells' | 'winners' | 'losers';

const FILTERS: { key: FilterType; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'buys', label: 'Buys' },
  { key: 'sells', label: 'Sells' },
  { key: 'winners', label: 'Winners' },
  { key: 'losers', label: 'Losers' },
];

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

function formatTime(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
    });
  } catch {
    return '--:--';
  }
}

export default function TradesScreen() {
  const [filter, setFilter] = useState<FilterType>('all');
  const { data: tradesData, refetch, isLoading } = useTrades();

  const handleFilterChange = (newFilter: FilterType) => {
    setFilter(newFilter);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const trades = tradesData?.trades || [];

  const filteredTrades = trades.filter((trade) => {
    switch (filter) {
      case 'buys':
        return trade.side === 'BUY';
      case 'sells':
        return trade.side === 'SELL';
      case 'winners':
        return (trade.pnl || 0) > 0;
      case 'losers':
        return (trade.pnl || 0) < 0;
      default:
        return true;
    }
  });

  const renderTrade = useCallback(
    ({ item }: { item: Trade }) => <TradeCard trade={item} />,
    []
  );

  const keyExtractor = useCallback((item: Trade) => item.id, []);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Trade History</Text>
      </View>

      {/* Section Header with Filter */}
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Recent Trades</Text>
        <Pressable style={styles.filterButton}>
          <Text style={styles.filterButtonText}>Filter</Text>
        </Pressable>
      </View>

      {/* Filter Chips */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.filterStrip}
        contentContainerStyle={styles.filterStripContent}
      >
        {FILTERS.map(({ key, label }) => (
          <Pressable
            key={key}
            onPress={() => handleFilterChange(key)}
            style={[styles.filterChip, filter === key && styles.filterChipActive]}
          >
            <Text
              style={[
                styles.filterChipText,
                filter === key && styles.filterChipTextActive,
              ]}
            >
              {label}
            </Text>
          </Pressable>
        ))}
      </ScrollView>

      {/* Trades List */}
      <FlatList
        data={filteredTrades}
        renderItem={renderTrade}
        keyExtractor={keyExtractor}
        style={styles.tradeList}
        contentContainerStyle={styles.tradeListContent}
        refreshControl={
          <RefreshControl
            refreshing={isLoading}
            onRefresh={refetch}
            tintColor={colors.text.secondary}
          />
        }
        ListEmptyComponent={
          <Card style={styles.emptyCard}>
            <View style={styles.emptyState}>
              <Ionicons
                name="document-text-outline"
                size={48}
                color={colors.text.muted}
              />
              <Text style={styles.emptyTitle}>No Trades Yet</Text>
              <Text style={styles.emptyText}>
                Completed trades will appear here
              </Text>
            </View>
          </Card>
        }
      />
    </SafeAreaView>
  );
}

function TradeCard({ trade }: { trade: Trade }) {
  const isBuy = trade.side === 'BUY';
  const pnl = trade.pnl || 0;
  const isWin = pnl > 0;
  const isLoss = pnl < 0;

  return (
    <Card style={styles.tradeCard}>
      <View style={styles.tradeLeft}>
        <View style={[styles.tradeType, isBuy ? styles.tradeTypeBuy : styles.tradeTypeSell]}>
          <Ionicons
            name={isBuy ? 'arrow-up' : 'arrow-down'}
            size={16}
            color={isBuy ? colors.signal.gain : colors.signal.loss}
          />
        </View>
        <View>
          <Text style={styles.tradeSymbol}>{trade.symbol}</Text>
          <Text style={styles.tradeMeta}>
            {trade.quantity} @ ${trade.price.toFixed(2)}
          </Text>
        </View>
      </View>
      <View style={styles.tradeRight}>
        <Text
          style={[
            styles.tradePnl,
            isWin && styles.tradePnlUp,
            isLoss && styles.tradePnlDown,
          ]}
        >
          {formatCurrency(pnl, true)}
        </Text>
        <Text style={styles.tradeTime}>{formatTime(trade.timestamp)}</Text>
      </View>
    </Card>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.deep,
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
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 8,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  filterButton: {
    paddingHorizontal: 8,
    paddingVertical: 4,
  },
  filterButtonText: {
    fontSize: 12,
    fontWeight: '500',
    color: colors.signal.active,
  },
  filterStrip: {
    marginBottom: 12,
  },
  filterStripContent: {
    paddingHorizontal: 20,
    gap: 6,
  },
  filterChip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: colors.bg.elevated,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  filterChipActive: {
    backgroundColor: colors.bg.deep,
    borderColor: colors.border,
  },
  filterChipText: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.text.tertiary,
  },
  filterChipTextActive: {
    color: colors.text.primary,
  },
  tradeList: {
    flex: 1,
  },
  tradeListContent: {
    paddingHorizontal: 20,
    paddingBottom: 20,
    gap: 8,
  },
  emptyCard: {
    marginTop: 20,
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
  tradeCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
  },
  tradeLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  tradeType: {
    width: 32,
    height: 32,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tradeTypeBuy: {
    backgroundColor: colors.signal.gainDim,
  },
  tradeTypeSell: {
    backgroundColor: colors.signal.lossDim,
  },
  tradeSymbol: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.primary,
  },
  tradeMeta: {
    fontSize: 10,
    color: colors.text.muted,
    marginTop: 1,
  },
  tradeRight: {
    alignItems: 'flex-end',
  },
  tradePnl: {
    fontSize: 13,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.secondary,
  },
  tradePnlUp: {
    color: colors.signal.gain,
  },
  tradePnlDown: {
    color: colors.signal.loss,
  },
  tradeTime: {
    fontSize: 10,
    color: colors.text.muted,
    marginTop: 2,
  },
});
