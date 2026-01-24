import { View, Text, StyleSheet, ViewStyle } from 'react-native';
import { colors } from '../../lib/constants';

type BadgeVariant = 'gain' | 'loss' | 'warning' | 'active' | 'purple' | 'neutral';

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  style?: ViewStyle;
}

const variantStyles: Record<BadgeVariant, { bg: string; text: string }> = {
  gain: { bg: colors.signal.gainDim, text: colors.signal.gainBright },
  loss: { bg: colors.signal.lossDim, text: colors.signal.lossBright },
  warning: { bg: colors.signal.warnDim, text: colors.signal.warning },
  active: { bg: colors.signal.activeDim, text: colors.signal.active },
  purple: { bg: colors.signal.purpleDim, text: colors.signal.purple },
  neutral: { bg: colors.bg.elevated, text: colors.text.secondary },
};

export function Badge({ variant = 'neutral', children, style }: BadgeProps) {
  const variantStyle = variantStyles[variant];

  return (
    <View style={[styles.badge, { backgroundColor: variantStyle.bg }, style]}>
      <Text style={[styles.text, { color: variantStyle.text }]}>{children}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 6,
    alignSelf: 'flex-start',
  },
  text: {
    fontSize: 12,
    fontWeight: '600',
    fontFamily: 'JetBrains Mono',
  },
});
