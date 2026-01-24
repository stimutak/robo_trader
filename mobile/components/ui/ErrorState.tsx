import { View, Text, Pressable, StyleSheet, ViewStyle } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import * as Haptics from 'expo-haptics';
import { colors } from '../../lib/constants';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  style?: ViewStyle;
  compact?: boolean;
}

export function ErrorState({
  title = 'Connection Error',
  message = 'Unable to load data. Check your connection and try again.',
  onRetry,
  style,
  compact = false,
}: ErrorStateProps) {
  const handleRetry = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    onRetry?.();
  };

  if (compact) {
    return (
      <View style={[styles.compactContainer, style]}>
        <Ionicons name="cloud-offline-outline" size={20} color={colors.signal.loss} />
        <Text style={styles.compactText}>{title}</Text>
        {onRetry && (
          <Pressable onPress={handleRetry} style={styles.compactRetry}>
            <Text style={styles.compactRetryText}>Retry</Text>
          </Pressable>
        )}
      </View>
    );
  }

  return (
    <View style={[styles.container, style]}>
      <View style={styles.iconContainer}>
        <Ionicons name="cloud-offline-outline" size={48} color={colors.signal.loss} />
      </View>
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.message}>{message}</Text>
      {onRetry && (
        <Pressable
          onPress={handleRetry}
          style={({ pressed }) => [styles.retryButton, pressed && styles.retryButtonPressed]}
        >
          <Ionicons name="refresh" size={16} color={colors.text.primary} />
          <Text style={styles.retryText}>Try Again</Text>
        </Pressable>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    paddingVertical: 40,
    paddingHorizontal: 20,
  },
  iconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: colors.signal.lossDim,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 17,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: 8,
  },
  message: {
    fontSize: 14,
    color: colors.text.secondary,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 24,
  },
  retryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 20,
    paddingVertical: 12,
    backgroundColor: colors.bg.elevated,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
  },
  retryButtonPressed: {
    backgroundColor: colors.bg.surface,
    transform: [{ scale: 0.98 }],
  },
  retryText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.primary,
  },
  // Compact styles
  compactContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: colors.signal.lossDim,
    borderRadius: 10,
  },
  compactText: {
    flex: 1,
    fontSize: 13,
    color: colors.signal.loss,
  },
  compactRetry: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: colors.bg.surface,
    borderRadius: 6,
  },
  compactRetryText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.text.primary,
  },
});
