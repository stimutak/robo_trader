#!/usr/bin/env python3
"""Fix column name issues in ml_features.py"""

import re

# Read the file
with open('robo_trader/features/ml_features.py', 'r') as f:
    content = f.read()

# Replace all column references to handle both uppercase and lowercase
replacements = [
    (r"df\['close'\]", "df[close_col]"),
    (r"df\['high'\]", "df[high_col]"),
    (r"df\['low'\]", "df[low_col]"),
    (r"df\['open'\]", "df[open_col]"),
    (r"df\['volume'\]", "df[volume_col]"),
]

# Apply replacements
for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Add column detection before each method that uses df
methods_using_df = [
    'def detect_market_regime',
    'def calculate_sentiment_features',
    'def calculate_microstructure_features'
]

for method in methods_using_df:
    # Find the method and add column detection at the beginning
    pattern = f"({method}.*?\\n.*?try:)"
    def add_column_detection(match):
        return match.group(1) + """
            # Handle both lowercase and uppercase column names
            close_col = 'close' if 'close' in df.columns else 'Close'
            high_col = 'high' if 'high' in df.columns else 'High'
            low_col = 'low' if 'low' in df.columns else 'Low'
            open_col = 'open' if 'open' in df.columns else 'Open'
            volume_col = 'volume' if 'volume' in df.columns else 'Volume'"""
    
    content = re.sub(pattern, add_column_detection, content, flags=re.DOTALL)

# Write the fixed file
with open('robo_trader/features/ml_features.py', 'w') as f:
    f.write(content)

print("Fixed ml_features.py column references")