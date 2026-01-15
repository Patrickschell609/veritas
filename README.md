# PROJECT VERITAS

## Forensic Deepfake Detection Tool

Analyzes video for visual manipulation, audio artifacts, and scam indicators.
Outputs annotated video with frame-by-frame proof.

### How Detection Actually Works

Most deepfake detectors fail because they rely on single signals. VERITAS uses multiple independent detection vectors:

**1. Laplacian Variance (Face Texture Analysis)**
- Real faces have natural texture variation (pores, fine lines, micro-shadows)
- Deepfakes often produce unnaturally smooth faces
- We measure texture variance using Laplacian filters
- Variance < 120 = suspicious smoothness

**2. Boundary Consistency Analysis**
- When a face is composited onto video, the edges often don't match
- Color temperature, lighting direction, and skin tone differ at boundaries
- We compare face region color to surrounding region
- Large difference = likely composite

**3. Spectral Audio Analysis**
- Real human speech has rich high-frequency content
- Voice synthesis often lacks natural high-frequency harmonics
- We use FFT to analyze frequency distribution
- Missing high-frequency energy = synthetic voice

**4. Context/Scam Detection**
- OCR extracts text from video frames
- Pattern matching for wallet addresses (ETH, BTC)
- Keyword detection for common scam phrases
- "Send ETH", "giveaway", "double your crypto" = scam indicators

### Why Most Tools Are Unreliable

Coffeezilla is right - most detection tools ARE unreliable because:

1. **Single-vector detection** - Easy to bypass by improving one aspect
2. **Training data bias** - Models trained on old deepfakes miss new ones
3. **Confidence theater** - High percentages with no real basis
4. **No explainability** - "It's fake" with no proof

VERITAS provides:
- Multiple independent detection methods
- Per-frame analysis with visible metrics
- Specific indicators (texture variance numbers, frequency analysis)
- Annotated output video showing exactly what triggered detection

### Requirements

**System:**
- ffmpeg
- yt-dlp (for URL downloads)
- tesseract-ocr
- OpenCV

**Install on Ubuntu/Debian/Kali:**
```bash
sudo apt update
sudo apt install ffmpeg tesseract-ocr libtesseract-dev libopencv-dev clang libclang-dev
pip install yt-dlp
```

### Usage

**1. Analyze video:**
```bash
cd veritas-core
cargo build --release
./target/release/veritas-core "https://youtube.com/watch?v=XXXXX"
# or local file:
./target/release/veritas-core /path/to/video.mp4
```

**2. Generate annotated output:**
```bash
cd veritas-viz
pip install -r requirements.txt
python heatmap_overlay.py
```

**3. View results:**
- `veritas_report.json` - Full analysis report
- `veritas_output.mp4` - Annotated video with overlays

### Detection Methods

| Method | What It Detects | How |
|--------|-----------------|-----|
| Laplacian Variance | Unnaturally smooth faces | Texture analysis via edge detection |
| Boundary Analysis | Face composite artifacts | Color consistency at face edges |
| Spectral Analysis | Synthetic voice | FFT frequency distribution |
| OCR + Keywords | Scam content | Text extraction + pattern matching |

### Output

The annotated video includes:
- Real-time metrics overlay (top banner)
- Face detection with per-face analysis
- Suspicious regions highlighted in red with scan lines
- Texture variance displayed per face
- Overall verdict banner (bottom)

### Limitations

This tool is NOT perfect. It CAN:
- Detect common deepfake artifacts
- Flag suspicious content for human review
- Provide evidence for investigation

It CANNOT:
- Guarantee 100% accuracy
- Detect highly sophisticated deepfakes
- Replace human judgment

Use as ONE tool in your investigation toolkit, not as sole evidence.

### For Journalists

When reporting on suspected deepfakes:
1. Run VERITAS analysis
2. Document the specific indicators flagged
3. Cross-reference with other evidence
4. Contact subjects for verification
5. Consult with forensic experts

The annotated video can be used to explain to viewers exactly WHY content is suspected to be fake.

---

Built for investigative journalism. Use responsibly.

Contact: @CtrlAlt8080
