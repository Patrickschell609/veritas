#!/bin/bash
# Launch VERITAS GUI
cd "$(dirname "$0")"

# Setup venv if needed
if [ ! -d ".venv" ]; then
    echo "First run - setting up..."
    python3 -m venv .venv
    .venv/bin/pip install opencv-python numpy yt-dlp --quiet
    echo "Setup complete!"
fi

.venv/bin/python veritas_gui.py
