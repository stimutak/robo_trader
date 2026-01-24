import FontAwesome from '@expo/vector-icons/FontAwesome';
import { DarkTheme, ThemeProvider } from '@react-navigation/native';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { useEffect } from 'react';
import { StatusBar, View } from 'react-native';
import 'react-native-reanimated';
import { colors } from '../lib/constants';

export {
  ErrorBoundary,
} from 'expo-router';

export const unstable_settings = {
  initialRouteName: '(tabs)',
};

SplashScreen.preventAutoHideAsync();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      staleTime: 5000,
    },
  },
});

// Custom dark theme for RoboTrader
const RoboTraderDarkTheme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    primary: colors.signal.active,
    background: colors.bg.deep,
    card: colors.bg.surface,
    text: colors.text.primary,
    border: colors.border,
    notification: colors.signal.warning,
  },
};

export default function RootLayout() {
  const [loaded, error] = useFonts({
    'JetBrains Mono': require('../assets/fonts/SpaceMono-Regular.ttf'),
    ...FontAwesome.font,
  });

  useEffect(() => {
    if (error) throw error;
  }, [error]);

  useEffect(() => {
    if (loaded) {
      SplashScreen.hideAsync();
    }
  }, [loaded]);

  if (!loaded) {
    return null;
  }

  return <RootLayoutNav />;
}

function RootLayoutNav() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider value={RoboTraderDarkTheme}>
        <StatusBar barStyle="light-content" />
        <View style={{ flex: 1, backgroundColor: colors.bg.deep }}>
          <Stack>
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen
              name="position/[symbol]"
              options={{
                presentation: 'card',
                headerShown: true,
                headerStyle: { backgroundColor: colors.bg.surface },
                headerTintColor: colors.text.primary,
              }}
            />
          </Stack>
        </View>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
