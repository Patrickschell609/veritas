# PROJECT VERITAS

## Forensic Deepfake Detection Tool

Multi-vector deepfake analysis for investigative journalists. Built for reliability, explainability, and ease of use.

![Version](https://img.shields.io/badge/version-2.0.0-cyan)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Quick Start

```bash
# 1. Install (one time)
chmod +x install.sh
./install.sh

# 2. Analyze a video
./run.sh https://youtube.com/watch?v=VIDEO_ID
# or
./run.sh suspicious_video.mp4

# 3. Or use the GUI
./VERITAS.sh
```

---

## What It Does

VERITAS analyzes videos using **8 independent detection methods**:

| Method | What It Detects | How |
|--------|-----------------|-----|
| **Laplacian Variance** | Unnaturally smooth faces | Texture analysis - deepfakes often lack natural skin texture |
| **Boundary Consistency** | Face composite artifacts | Color/lighting mismatch at face edges |
| **Spectral Audio** | Synthetic voice | FFT analysis - AI voices lack high-frequency harmonics |
| **Blink Detection** | Unnatural blink patterns | Deepfakes often don't blink correctly |
| **Noise Pattern** | GAN fingerprints | AI-generated content has uniform noise |
| **Entropy Analysis** | Abnormal randomness | Synthetic content has unusual entropy distribution |
| **Temporal Consistency** | Face position jumps | Faces shouldn't teleport between frames |
| **OCR + Keywords** | Scam content | Wallet addresses, "giveaway" phrases |

### Why Multiple Methods?

Most deepfake detectors fail because they rely on **single signals** that can be bypassed. VERITAS combines independent methods - even if one fails, others can still detect manipulation.

---

## Output

After analysis, you get:

1. **veritas_report.json** - Machine-readable data with all metrics
2. **veritas_report.html** - Beautiful shareable report for articles/videos
3. **veritas_output.mp4** - Annotated video showing what triggered detection

### Confidence Levels

| Score | Meaning |
|-------|---------|
| **>75%** | HIGH - Almost certainly fake/manipulated |
| **50-75%** | MEDIUM - Suspicious, needs human review |
| **<50%** | LOW - Likely authentic |

---

## Installation

### Automatic (Recommended)

```bash
chmod +x install.sh
./install.sh
```

The installer automatically detects your OS and installs everything needed.

### Manual

**macOS:**
```bash
brew install python3 ffmpeg tesseract
python3 -m venv .venv
.venv/bin/pip install opencv-python numpy yt-dlp pillow
```

**Ubuntu/Debian/Kali:**
```bash
sudo apt install python3 python3-venv ffmpeg tesseract-ocr libgl1-mesa-glx
python3 -m venv .venv
.venv/bin/pip install opencv-python numpy yt-dlp pillow
```

---

## Usage

### Command Line

```bash
# Analyze YouTube video
./run.sh https://youtube.com/watch?v=ABC123

# Analyze local file
./run.sh /path/to/video.mp4

# Fast analysis (skip video rendering)
./run.sh video.mp4 --no-video

# JSON only (minimal output)
./run.sh video.mp4 --json-only
```

### GUI

```bash
./VERITAS.sh
```

Or double-click `VERITAS.sh` in your file manager.

### Python API

```python
from veritas import analyze

report = analyze("video.mp4", generate_video=True, generate_html=True)

print(f"Confidence: {report.confidence * 100:.0f}%")
print(f"Verdict: {report.verdict}")

for flag in report.flags:
    print(f"  ! {flag}")
```

---

## For Journalists

When reporting on suspected deepfakes:

1. **Run VERITAS** - Get the initial analysis
2. **Document specific indicators** - Note which methods flagged issues
3. **Cross-reference** - Compare with known authentic footage
4. **Verify chain of custody** - Where did the video come from?
5. **Consult experts** - VERITAS is one tool, not definitive proof

The **HTML report** can be shared with editors or used to explain findings to viewers.

---

## Limitations

VERITAS is a **detection aid**, not a lie detector.

**It CAN:**
- Detect common deepfake artifacts
- Flag suspicious content for review
- Provide evidence for investigation
- Explain why content was flagged

**It CANNOT:**
- Guarantee 100% accuracy
- Detect highly sophisticated deepfakes
- Replace human judgment
- Serve as legal proof

Advanced deepfakes (especially those with post-processing) may evade detection. Always use VERITAS as **one tool in your toolkit**, not the only one.

---

## Technical Details

### Detection Thresholds

```python
THRESHOLDS = {
    "laplacian_variance": 100,    # Below = suspiciously smooth
    "boundary_distance": 35,      # Above = color mismatch
    "high_freq_ratio": 0.03,      # Below = synthetic voice
    "blink_rate_min": 0.1,        # Below = not blinking enough
    "blink_rate_max": 0.5,        # Above = blinking too much
}
```

### Confidence Calculation

```
confidence = (
    visual_score * 0.30 +
    audio_score * 0.20 +
    scam_score * 0.20 +
    entropy_score * 0.10 +
    blink_score * 0.10 +
    noise_score * 0.10
)
```

Multiple strong signals boost confidence (if 3+ methods flag >50%, confidence is boosted by 20%).

---

## Contributing

Pull requests welcome. Priority areas:

- [ ] Lip sync detection (audio-visual correlation)
- [ ] Better temporal analysis (optical flow)
- [ ] Model-based detection (ResNet/EfficientNet)
- [ ] Windows native support
- [ ] Batch processing mode

---

## License

MIT - Use freely for journalism, research, and education.

---

## Contact

Questions or feedback: @CtrlAlt8080

---

*Built for investigative journalism. Use responsibly.*
