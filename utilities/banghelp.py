#!/usr/bin/env python3
"""
Helper script to display help for the CWT tool.
This is an alias for 'python cwt.py help'.
"""

import sys
import subprocess

# Construct the help command
cmd = [sys.executable, "cwt.py", "help"]

# If there are additional arguments, pass them along
if len(sys.argv) > 1:
    cmd.extend(sys.argv[1:])

# Run the help command
subprocess.run(cmd) 