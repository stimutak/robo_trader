import { useState, memo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
} from 'react-native';
import * as Clipboard from 'expo-clipboard';
import * as Haptics from 'expo-haptics';
import { LogEntry as LogEntryType } from '../../lib/types';
import { colors, logLevelColors, logLevelBgColors } from '../../lib/constants';

interface LogEntryProps {
  log: LogEntryType;
  searchQuery?: string;
}

function highlightText(text: string, query: string): React.ReactNode {
  if (!query) return text;

  const parts = text.split(new RegExp(`(${query})`, 'gi'));
  return parts.map((part, i) =>
    part.toLowerCase() === query.toLowerCase() ? (
      <Text key={i} style={styles.highlight}>
        {part}
      </Text>
    ) : (
      part
    )
  );
}

function formatTime(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    });
  } catch {
    return '--:--:--';
  }
}

function LogEntryComponent({ log, searchQuery = '' }: LogEntryProps) {
  const [expanded, setExpanded] = useState(false);

  const handlePress = () => {
    if (log.context) {
      setExpanded(!expanded);
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    }
  };

  const handleLongPress = async () => {
    await Clipboard.setStringAsync(log.message);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const levelColor = logLevelColors[log.level] || colors.text.secondary;
  const levelBgColor = logLevelBgColors[log.level] || colors.bg.elevated;

  return (
    <Pressable
      onPress={handlePress}
      onLongPress={handleLongPress}
      style={({ pressed }) => [
        styles.container,
        pressed && styles.pressed,
        expanded && styles.expanded,
      ]}
    >
      <View style={styles.meta}>
        <Text style={styles.time}>{formatTime(log.timestamp)}</Text>
        <View style={[styles.levelBadge, { backgroundColor: levelBgColor }]}>
          <Text style={[styles.levelText, { color: levelColor }]}>
            {log.level}
          </Text>
        </View>
        <Text style={styles.source}>{log.source}</Text>
      </View>
      <Text style={styles.message}>
        {highlightText(log.message, searchQuery)}
      </Text>
      {expanded && log.context && (
        <View style={styles.contextContainer}>
          <Text style={styles.context}>
            {JSON.stringify(log.context, null, 2)}
          </Text>
        </View>
      )}
    </Pressable>
  );
}

export const LogEntryMemo = memo(LogEntryComponent);
export { LogEntryComponent as LogEntry };

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.02)',
  },
  pressed: {
    backgroundColor: colors.bg.surface,
  },
  expanded: {
    backgroundColor: colors.bg.surface,
  },
  meta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 3,
  },
  time: {
    fontSize: 10,
    fontFamily: 'JetBrains Mono',
    color: colors.text.muted,
  },
  levelBadge: {
    paddingHorizontal: 5,
    paddingVertical: 1,
    borderRadius: 3,
  },
  levelText: {
    fontSize: 9,
    fontWeight: '600',
    fontFamily: 'JetBrains Mono',
    textTransform: 'uppercase',
  },
  source: {
    fontSize: 10,
    fontFamily: 'JetBrains Mono',
    color: colors.signal.purple,
  },
  message: {
    fontSize: 11,
    fontFamily: 'JetBrains Mono',
    color: colors.text.secondary,
    lineHeight: 16,
  },
  highlight: {
    backgroundColor: colors.signal.warning,
    color: colors.bg.deep,
    paddingHorizontal: 2,
    borderRadius: 2,
  },
  contextContainer: {
    marginTop: 8,
    padding: 10,
    backgroundColor: colors.bg.elevated,
    borderRadius: 6,
  },
  context: {
    fontSize: 10,
    fontFamily: 'JetBrains Mono',
    color: colors.text.tertiary,
  },
});
