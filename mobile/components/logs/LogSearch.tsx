import { View, TextInput, StyleSheet } from 'react-native';
import { colors } from '../../lib/constants';
import { useLogsStore } from '../../stores/logs';

export function LogSearch() {
  const search = useLogsStore((s) => s.search);
  const setSearch = useLogsStore((s) => s.setSearch);

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Search logs..."
        placeholderTextColor={colors.text.muted}
        value={search}
        onChangeText={setSearch}
        autoCapitalize="none"
        autoCorrect={false}
        clearButtonMode="while-editing"
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  input: {
    backgroundColor: colors.bg.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 13,
    fontFamily: 'JetBrains Mono',
    color: colors.text.primary,
  },
});
