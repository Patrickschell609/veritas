# VERITAS - Quick Start

## What It Does
Analyzes videos for deepfake indicators. Outputs a report + annotated video showing what's suspicious.

## Setup (One Time)
```bash
# Install system deps (Linux/Mac)
sudo apt install ffmpeg tesseract-ocr   # Linux
brew install ffmpeg tesseract           # Mac

# Make run script executable
chmod +x run.sh
```

## Usage
```bash
# Analyze a YouTube video
./run.sh "https://youtube.com/watch?v=XXXXX"

# Analyze a local file
./run.sh suspicious_video.mp4
```

## Output

**veritas_report.json** - Raw data:
```json
{
  "confidence": 0.85,
  "verdict": "HIGH CONFIDENCE: SYNTHETIC/MANIPULATED CONTENT",
  "metrics": {
    "visual_manipulation": 0.72,
    "audio_artifacts": 0.91,
    "scam_indicators": 0.60
  }
}
```

**veritas_output.mp4** - Annotated video with:
- Face detection boxes (green = OK, red = suspicious)
- Real-time metrics overlay
- Verdict banner

## What The Scores Mean

| Score | What It Measures | Red Flag If... |
|-------|------------------|----------------|
| Visual Manipulation | Face texture smoothness, edge artifacts | > 50% |
| Audio Artifacts | Missing high frequencies in voice | > 50% |
| Scam Indicators | Wallet addresses, "giveaway" keywords | > 30% |
| Entropy Anomaly | Unnatural randomness patterns | > 50% |

## Confidence Levels

- **> 75%** = HIGH - Almost certainly fake/manipulated
- **50-75%** = MEDIUM - Suspicious, needs human review
- **< 50%** = LOW - Probably authentic

## Limitations

This tool helps identify suspicious content. It is NOT:
- 100% accurate
- Legal proof
- A replacement for human judgment

Use it as ONE input in your investigation, not the only one.

## Questions?
Contact: @CtrlAlt8080
