#!/bin/bash
# VERITAS - One-click deepfake detection
# Usage: ./run.sh <video_url_or_file>

cd "$(dirname "$0")"

if [ -z "$1" ]; then
    echo ""
    echo "  VERITAS - Deepfake Detection Tool"
    echo "  =================================="
    echo ""
    echo "  Usage:"
    echo "    ./run.sh https://youtube.com/watch?v=VIDEO_ID"
    echo "    ./run.sh /path/to/video.mp4"
    echo ""
    echo "  Output:"
    echo "    veritas_report.json  - Analysis data"
    echo "    veritas_output.mp4   - Annotated video"
    echo ""
    exit 1
fi

# Check dependencies
if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not installed. Run: sudo apt install ffmpeg"
    exit 1
fi

# Setup venv if needed
if [ ! -d ".venv" ]; then
    echo "[*] First run - setting up environment..."
    python3 -m venv .venv
    .venv/bin/pip install opencv-python numpy yt-dlp --quiet
    echo "[*] Setup complete!"
fi

# Run analysis
echo ""
.venv/bin/python veritas.py "$1"
echo ""
echo "Done! Check veritas_report.json and veritas_output.mp4"
