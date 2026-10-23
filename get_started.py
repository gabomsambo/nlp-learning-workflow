#!/usr/bin/env python
"""Superseded by ./workstation.

The old checker looked for the hosted Supabase project, which is gone.
Start the app with ./workstation instead.
"""

import sys


def main() -> int:
    """Print the workstation command and exit."""
    print("get_started.py is superseded by ./workstation.")
    print("Start the workstation with:")
    print("  ./workstation up")
    print("IndexTTS stays off unless you pass --tts:")
    print("  ./workstation up --tts")
    print("Run ./workstation with no arguments for status, logs, and stopping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
