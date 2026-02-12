#!/usr/bin/env python3
"""
Emoji checker hook for Claude Code.
Detects emojis in edited files and asks Claude to remove them.
"""

import json
import sys
import os
import re

# Emoji pattern to detect any emoji
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U00002600-\U000026ff"  # misc symbols
    "\U0001fa70-\U0001faff"  # symbols and pictographs extended-A
    "\U00002300-\U000023ff"  # misc technical
    "]+",
    flags=re.UNICODE,
)

try:
    input_data = json.load(sys.stdin)
    tool_input = input_data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path or not os.path.exists(file_path):
        sys.exit(0)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check for emojis
    emojis_found = EMOJI_PATTERN.findall(content)

    if emojis_found:
        # Exit with code 2 to block and provide feedback to Claude
        print(
            f"Emojis are not allowed in files. Please remove or replace the emojis in {file_path} with text equivalents like [X], [OK], [WARNING], etc.",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)

except:
    sys.exit(0)
