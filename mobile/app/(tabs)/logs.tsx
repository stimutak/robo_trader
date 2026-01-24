import { useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Pressable,
  SafeAreaView,
} from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import * as Haptics from 'expo-haptics';
import { colors } from '../../lib/constants';
import { useLogsStore } from '../../stores/logs';
import { useWebSocket } from '../../hooks/useWebSocket';
import { LogEntryMemo } from '../../components/logs/LogEntry';
import { LogFilter } from '../../components/logs/LogFilter';
import { LogSearch } from '../../components/logs/LogSearch';
import { LogEntry } from '../../lib/types';

export default function LogsScreen() {
  const flatListRef = useRef<FlatList>(null);

  const logs = useLogsStore((s) => s.logs);
  const filter = useLogsStore((s) => s.filter);
  const search = useLogsStore((s) => s.search);
  const autoScroll = useLogsStore((s) => s.autoScroll);
  const wsConnected = useLogsStore((s) => s.wsConnected);
  const toggleAutoScroll = useLogsStore((s) => s.toggleAutoScroll);
  const setAutoScroll = useLogsStore((s) => s.setAutoScroll);
  const clear = useLogsStore((s) => s.clear);

  // Initialize WebSocket connection
  useWebSocket();

  // Filter logs based on current filter and search
  const filteredLogs = logs.filter((log) => {
    if (filter !== 'all' && log.level !== filter) return false;
    if (search && !log.message.toLowerCase().includes(search.toLowerCase())) {
      return false;
    }
    return true;
  });

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && filteredLogs.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [filteredLogs.length, autoScroll]);

  const handleAutoScrollToggle = () => {
    toggleAutoScroll();
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const handleClear = () => {
    clear();
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const handleScroll = useCallback(
    (event: any) => {
      const { layoutMeasurement, contentOffset, contentSize } = event.nativeEvent;
      const isAtBottom =
        layoutMeasurement.height + contentOffset.y >= contentSize.height - 50;

      // Pause auto-scroll if user scrolls up
      if (!isAtBottom && autoScroll) {
        setAutoScroll(false);
      }
    },
    [autoScroll, setAutoScroll]
  );

  const renderLogEntry = useCallback(
    ({ item }: { item: LogEntry }) => (
      <LogEntryMemo log={item} searchQuery={search} />
    ),
    [search]
  );

  const keyExtractor = useCallback((item: LogEntry) => item.id, []);

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.title}>Logs</Text>
        <View style={styles.controls}>
          <Pressable
            onPress={handleAutoScrollToggle}
            style={[styles.button, autoScroll && styles.buttonActive]}
          >
            <Ionicons
              name="arrow-down"
              size={14}
              color={autoScroll ? colors.signal.active : colors.text.secondary}
            />
            <Text
              style={[
                styles.buttonText,
                autoScroll && styles.buttonTextActive,
              ]}
            >
              Auto
            </Text>
          </Pressable>
          <Pressable onPress={handleClear} style={styles.button}>
            <Ionicons name="trash-outline" size={14} color={colors.text.secondary} />
          </Pressable>
        </View>
      </View>

      {/* Connection Status */}
      <View
        style={[
          styles.connectionStatus,
          wsConnected ? styles.connected : styles.disconnected,
        ]}
      >
        <View
          style={[
            styles.statusDot,
            { backgroundColor: wsConnected ? colors.signal.gain : colors.signal.loss },
          ]}
        />
        <Text style={styles.connectionText}>
          {wsConnected ? 'Connected to log stream' : 'Disconnected - reconnecting...'}
        </Text>
      </View>

      {/* Filters */}
      <LogFilter />

      {/* Search */}
      <LogSearch />

      {/* Log List */}
      <FlatList
        ref={flatListRef}
        data={filteredLogs}
        renderItem={renderLogEntry}
        keyExtractor={keyExtractor}
        style={styles.logList}
        contentContainerStyle={styles.logListContent}
        onScroll={handleScroll}
        scrollEventThrottle={16}
        initialNumToRender={20}
        maxToRenderPerBatch={20}
        windowSize={10}
        removeClippedSubviews={true}
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Ionicons
              name="document-text-outline"
              size={48}
              color={colors.text.muted}
            />
            <Text style={styles.emptyTitle}>No Logs</Text>
            <Text style={styles.emptyText}>
              Logs will appear here when the trading system is running
            </Text>
          </View>
        }
      />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg.deep,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.primary,
  },
  controls: {
    flexDirection: 'row',
    gap: 8,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 6,
    backgroundColor: colors.bg.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 6,
  },
  buttonActive: {
    backgroundColor: colors.signal.activeDim,
    borderColor: colors.signal.active,
  },
  buttonText: {
    fontSize: 11,
    fontWeight: '500',
    color: colors.text.secondary,
  },
  buttonTextActive: {
    color: colors.signal.active,
  },
  connectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  connected: {
    backgroundColor: 'transparent',
  },
  disconnected: {
    backgroundColor: colors.signal.lossDim,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  connectionText: {
    fontSize: 11,
    color: colors.text.tertiary,
  },
  logList: {
    flex: 1,
  },
  logListContent: {
    paddingBottom: 20,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
    paddingHorizontal: 40,
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
    lineHeight: 20,
  },
});
