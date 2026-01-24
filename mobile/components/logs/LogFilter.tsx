import { View, Text, StyleSheet, ScrollView, Pressable } from 'react-native';
import * as Haptics from 'expo-haptics';
import { colors, logLevelColors } from '../../lib/constants';
import { LogLevel, useLogsStore } from '../../stores/logs';

const LEVELS: { key: LogLevel; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'DEBUG', label: 'Debug' },
  { key: 'INFO', label: 'Info' },
  { key: 'WARNING', label: 'Warn' },
  { key: 'ERROR', label: 'Error' },
];

export function LogFilter() {
  const filter = useLogsStore((s) => s.filter);
  const setFilter = useLogsStore((s) => s.setFilter);

  const handlePress = (level: LogLevel) => {
    setFilter(level);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {LEVELS.map(({ key, label }) => {
          const isActive = filter === key;
          const levelColor =
            key !== 'all' ? logLevelColors[key] : colors.text.primary;

          return (
            <Pressable
              key={key}
              onPress={() => handlePress(key)}
              style={[
                styles.chip,
                isActive && styles.chipActive,
              ]}
            >
              <Text
                style={[
                  styles.chipText,
                  isActive && { color: levelColor },
                ]}
              >
                {label}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: colors.bg.surface,
  },
  scrollContent: {
    paddingHorizontal: 20,
    gap: 6,
  },
  chip: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: colors.bg.elevated,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  chipActive: {
    backgroundColor: colors.bg.deep,
    borderColor: colors.border,
  },
  chipText: {
    fontSize: 10,
    fontWeight: '600',
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 0.4,
  },
});
