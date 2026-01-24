import { View, StyleSheet } from 'react-native';
import { colors } from '../../lib/constants';

type StatusDotVariant = 'ok' | 'warning' | 'error' | 'off';

interface StatusDotProps {
  variant?: StatusDotVariant;
  size?: number;
  pulse?: boolean;
}

const variantColors: Record<StatusDotVariant, string> = {
  ok: colors.signal.gain,
  warning: colors.signal.warning,
  error: colors.signal.loss,
  off: colors.text.tertiary,
};

export function StatusDot({ variant = 'off', size = 6, pulse = false }: StatusDotProps) {
  const color = variantColors[variant];

  return (
    <View
      style={[
        styles.dot,
        {
          width: size,
          height: size,
          backgroundColor: color,
        },
        variant === 'ok' && styles.glow,
      ]}
    />
  );
}

const styles = StyleSheet.create({
  dot: {
    borderRadius: 100,
  },
  glow: {
    shadowColor: colors.signal.gain,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 4,
    elevation: 4,
  },
});
