import { View, ViewProps, StyleSheet } from 'react-native';
import { colors, spacing } from '../../lib/constants';

interface CardProps extends ViewProps {
  variant?: 'surface' | 'elevated';
}

export function Card({ variant = 'surface', style, children, ...props }: CardProps) {
  return (
    <View
      style={[
        styles.card,
        variant === 'elevated' && styles.elevated,
        style,
      ]}
      {...props}
    >
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bg.surface,
    borderRadius: spacing.cardRadius,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.cardPadding,
  },
  elevated: {
    backgroundColor: colors.bg.elevated,
  },
});
