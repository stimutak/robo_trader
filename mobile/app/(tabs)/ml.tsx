import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  SafeAreaView,
} from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { colors } from '../../lib/constants';
import { useMLStatus, usePredictions } from '../../hooks/useAPI';
import { Card } from '../../components/ui/Card';
import { Prediction } from '../../lib/types';

export default function MLScreen() {
  const { data: mlData, refetch: refetchML, isLoading } = useMLStatus();
  const { data: predsData, refetch: refetchPreds } = usePredictions();

  const handleRefresh = async () => {
    await Promise.all([refetchML(), refetchPreds()]);
  };

  const predictions = predsData?.predictions || [];

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
          <Text style={styles.title}>ML Engine</Text>
        </View>

        {/* ML Status Card */}
        <Card style={styles.statusCard}>
          <View style={styles.statusTop}>
            <View style={styles.mlIndicator}>
              <View style={styles.mlIcon}>
                <Ionicons
                  name="hardware-chip-outline"
                  size={20}
                  color={colors.signal.purple}
                />
              </View>
              <View>
                <Text style={styles.mlTitle}>ML Enhanced</Text>
                <Text style={styles.mlSubtitle}>
                  {mlData?.models_trained ? 'Active' : 'No models trained'}
                </Text>
              </View>
            </View>
            <View style={styles.mlCount}>
              <Text style={styles.mlCountNum}>{mlData?.models_trained || 0}</Text>
              <Text style={styles.mlCountLabel}>Models</Text>
            </View>
          </View>
          <View style={styles.mlStats}>
            <View style={styles.mlStat}>
              <Text style={styles.mlStatVal}>{mlData?.feature_count || 0}</Text>
              <Text style={styles.mlStatLabel}>Features</Text>
            </View>
            <View style={styles.mlStat}>
              <Text style={styles.mlStatVal}>
                {mlData?.accuracy ? `${(mlData.accuracy * 100).toFixed(0)}%` : '--%'}
              </Text>
              <Text style={styles.mlStatLabel}>Accuracy</Text>
            </View>
            <View style={styles.mlStat}>
              <Text style={styles.mlStatVal}>
                {mlData?.last_training ? 'Today' : '--'}
              </Text>
              <Text style={styles.mlStatLabel}>Last Run</Text>
            </View>
          </View>
        </Card>

        {/* Section Header */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Current Predictions</Text>
        </View>

        {/* Predictions List */}
        {predictions.length === 0 ? (
          <Card style={styles.emptyCard}>
            <View style={styles.emptyState}>
              <Ionicons
                name="hardware-chip-outline"
                size={48}
                color={colors.text.muted}
              />
              <Text style={styles.emptyTitle}>No Predictions</Text>
              <Text style={styles.emptyText}>
                ML predictions will appear when models are running
              </Text>
            </View>
          </Card>
        ) : (
          <View style={styles.predictionList}>
            {predictions.slice(0, 10).map((pred) => (
              <PredictionCard key={pred.symbol} prediction={pred} />
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function PredictionCard({ prediction }: { prediction: Prediction }) {
  const signal = prediction.signal > 0 ? 'buy' : prediction.signal < 0 ? 'sell' : 'hold';
  const signalText = signal.toUpperCase();
  const confidence = (prediction.confidence || 0) * 100;

  const signalStyles = {
    buy: { bg: colors.signal.gainDim, color: colors.signal.gain },
    sell: { bg: colors.signal.lossDim, color: colors.signal.loss },
    hold: { bg: colors.bg.elevated, color: colors.text.tertiary },
  };

  const style = signalStyles[signal];

  return (
    <Card style={styles.predictionCard}>
      <View style={styles.predLeft}>
        <View style={[styles.predSignal, { backgroundColor: style.bg }]}>
          <Text style={[styles.predSignalText, { color: style.color }]}>
            {signalText}
          </Text>
        </View>
        <View>
          <Text style={styles.predSymbol}>{prediction.symbol}</Text>
          <Text style={styles.predModel}>{prediction.model || 'ML Enhanced'}</Text>
        </View>
      </View>
      <View style={styles.predConf}>
        <Text style={styles.predConfPct}>{confidence.toFixed(0)}%</Text>
        <Text style={styles.predConfLabel}>Confidence</Text>
      </View>
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
  statusCard: {
    marginHorizontal: 20,
    marginBottom: 16,
    padding: 16,
  },
  statusTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 14,
  },
  mlIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  mlIcon: {
    width: 40,
    height: 40,
    borderRadius: 10,
    backgroundColor: colors.signal.purpleDim,
    alignItems: 'center',
    justifyContent: 'center',
  },
  mlTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.primary,
  },
  mlSubtitle: {
    fontSize: 11,
    color: colors.text.tertiary,
    marginTop: 1,
  },
  mlCount: {
    alignItems: 'flex-end',
  },
  mlCountNum: {
    fontSize: 28,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.signal.purple,
  },
  mlCountLabel: {
    fontSize: 9,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
  },
  mlStats: {
    flexDirection: 'row',
    gap: 10,
  },
  mlStat: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 10,
    backgroundColor: colors.bg.elevated,
    borderRadius: 8,
  },
  mlStatVal: {
    fontSize: 15,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.primary,
    marginBottom: 2,
  },
  mlStatLabel: {
    fontSize: 9,
    fontWeight: '600',
    color: colors.text.muted,
    textTransform: 'uppercase',
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
  predictionList: {
    paddingHorizontal: 20,
    gap: 8,
  },
  predictionCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
  },
  predLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  predSignal: {
    width: 34,
    height: 34,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  predSignalText: {
    fontSize: 9,
    fontFamily: 'JetBrains Mono',
    fontWeight: '700',
  },
  predSymbol: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.primary,
  },
  predModel: {
    fontSize: 10,
    color: colors.text.muted,
    marginTop: 1,
  },
  predConf: {
    alignItems: 'flex-end',
  },
  predConfPct: {
    fontSize: 14,
    fontFamily: 'JetBrains Mono',
    fontWeight: '500',
    color: colors.text.primary,
  },
  predConfLabel: {
    fontSize: 9,
    color: colors.text.muted,
  },
});
