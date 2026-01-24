import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import { useLocalSearchParams, Stack } from 'expo-router';
import { colors } from '../../lib/constants';
import { usePositions, usePredictions, useTrades } from '../../hooks/useAPI';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';

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

export default function PositionDetailScreen() {
  const { symbol } = useLocalSearchParams<{ symbol: string }>();
  const { data: positionsData } = usePositions();
  const { data: predsData } = usePredictions();
  const { data: tradesData } = useTrades();

  const position = positionsData?.positions.find((p) => p.symbol === symbol);
  const prediction = predsData?.predictions.find((p) => p.symbol === symbol);
  const symbolTrades = tradesData?.trades.filter((t) => t.symbol === symbol) || [];

  if (!position) {
    return (
      <SafeAreaView style={styles.container}>
        <Stack.Screen options={{ title: symbol || 'Position' }} />
        <View style={styles.centered}>
          <Text style={styles.notFoundText}>Position not found</Text>
        </View>
      </SafeAreaView>
    );
  }

  const pnl = position.unrealized_pnl || 0;
  const pnlPct = position.unrealized_pnl_pct || 0;
  const value = position.market_value || position.quantity * position.current_price;
  const isWin = pnl > 0;
  const isLoss = pnl < 0;

  return (
    <SafeAreaView style={styles.container}>
      <Stack.Screen
        options={{
          title: symbol,
          headerStyle: { backgroundColor: colors.bg.surface },
          headerTintColor: colors.text.primary,
        }}
      />
      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Strategy */}
        <Text style={styles.sector}>{position.strategy || 'Unknown Strategy'}</Text>

        {/* P&L Hero */}
        <View style={styles.pnlHero}>
          <Text
            style={[
              styles.pnlValue,
              isWin && styles.pnlValueUp,
              isLoss && styles.pnlValueDown,
            ]}
          >
            {formatCurrency(pnl, true)}
          </Text>
          <Text
            style={[
              styles.pnlPct,
              isWin && styles.pnlPctUp,
              isLoss && styles.pnlPctDown,
            ]}
          >
            {formatPercent(pnlPct, true)}
          </Text>
        </View>

        {/* Position Details */}
        <Card style={styles.detailsCard}>
          <DetailRow label="Shares" value={position.quantity.toString()} />
          <DetailRow label="Avg Cost" value={`$${(position.entry_price || 0).toFixed(2)}`} />
          <DetailRow label="Current Price" value={`$${(position.current_price || 0).toFixed(2)}`} />
          <DetailRow label="Market Value" value={formatCurrency(value)} />
        </Card>

        {/* ML Prediction */}
        {prediction && (
          <>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>ML Prediction</Text>
            </View>
            <Card style={styles.predictionCard}>
              <View style={styles.predRow}>
                <Badge
                  variant={
                    prediction.signal > 0
                      ? 'gain'
                      : prediction.signal < 0
                      ? 'loss'
                      : 'neutral'
                  }
                >
                  {prediction.signal > 0 ? 'BUY' : prediction.signal < 0 ? 'SELL' : 'HOLD'}
                </Badge>
                <Text style={styles.predConfidence}>
                  {(prediction.confidence * 100).toFixed(0)}% confidence
                </Text>
              </View>
            </Card>
          </>
        )}

        {/* Recent Trades */}
        {symbolTrades.length > 0 && (
          <>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Recent Trades</Text>
            </View>
            <Card style={styles.tradesCard}>
              {symbolTrades.slice(0, 5).map((trade) => (
                <View key={trade.id} style={styles.tradeRow}>
                  <View style={styles.tradeLeft}>
                    <Text
                      style={[
                        styles.tradeSide,
                        trade.side === 'BUY' ? styles.tradeBuy : styles.tradeSell,
                      ]}
                    >
                      {trade.side}
                    </Text>
                    <Text style={styles.tradeQty}>
                      {trade.quantity} @ ${trade.price.toFixed(2)}
                    </Text>
                  </View>
                  <Text style={styles.tradeDate}>
                    {new Date(trade.timestamp).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                    })}
                  </Text>
                </View>
              ))}
            </Card>
          </>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={styles.detailValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.deep,
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  notFoundText: {
    fontSize: 16,
    color: colors.text.secondary,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 40,
  },
  sector: {
    fontSize: 14,
    color: colors.text.tertiary,
    marginBottom: 8,
  },
  pnlHero: {
    alignItems: 'center',
    marginBottom: 24,
  },
  pnlValue: {
    fontSize: 36,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.primary,
    marginBottom: 4,
  },
  pnlValueUp: {
    color: colors.signal.gain,
  },
  pnlValueDown: {
    color: colors.signal.loss,
  },
  pnlPct: {
    fontSize: 18,
    fontFamily: 'JetBrains Mono',
    color: colors.text.secondary,
  },
  pnlPctUp: {
    color: colors.signal.gain,
  },
  pnlPctDown: {
    color: colors.signal.loss,
  },
  detailsCard: {
    padding: 0,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  detailLabel: {
    fontSize: 14,
    color: colors.text.secondary,
  },
  detailValue: {
    fontSize: 14,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.primary,
  },
  sectionHeader: {
    marginTop: 24,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 11,
    fontWeight: '700',
    color: colors.text.muted,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  predictionCard: {
    padding: 16,
  },
  predRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  predConfidence: {
    fontSize: 14,
    color: colors.text.secondary,
  },
  tradesCard: {
    padding: 0,
  },
  tradeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  tradeLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  tradeSide: {
    fontSize: 11,
    fontWeight: '700',
    fontFamily: 'JetBrains Mono',
  },
  tradeBuy: {
    color: colors.signal.gain,
  },
  tradeSell: {
    color: colors.signal.loss,
  },
  tradeQty: {
    fontSize: 12,
    color: colors.text.secondary,
  },
  tradeDate: {
    fontSize: 11,
    color: colors.text.muted,
  },
});
