#!/usr/bin/env python3
"""
PROJECT VERITAS - Forensic Deepfake Detection Tool
Pure Python implementation for investigative journalism

Analyzes video for:
- Visual manipulation (face texture, boundary artifacts)
- Audio artifacts (spectral analysis)
- Scam indicators (OCR + pattern matching)

Output: JSON report + annotated video
"""

import cv2
import numpy as np
import json
import sys
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import wave
import struct

# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    visual_manipulation: float
    audio_artifacts: float
    scam_indicators: float
    temporal_inconsistency: float
    entropy_anomaly: float

@dataclass
class VeritasReport:
    target: str
    metrics: Metrics
    confidence: float
    verdict: str
    face_analysis: List[dict]
    flags: List[str]

# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def download_video(target: str) -> Tuple[str, str]:
    """Download video and extract audio."""
    video_out = "temp_video.mp4"
    audio_out = "temp_audio.wav"

    is_url = target.startswith(("http://", "https://", "www."))

    # Find yt-dlp (might be in venv)
    script_dir = Path(__file__).parent
    ytdlp_paths = [
        script_dir / ".venv" / "bin" / "yt-dlp",
        "yt-dlp",
        "/usr/bin/yt-dlp"
    ]
    ytdlp_cmd = "yt-dlp"
    for p in ytdlp_paths:
        if Path(p).exists():
            ytdlp_cmd = str(p)
            break

    if is_url:
        print("    Downloading from URL...")
        result = subprocess.run(
            [ytdlp_cmd, "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
             "-o", video_out, target],
            capture_output=True
        )
        if result.returncode != 0:
            print("    yt-dlp failed, trying curl...")
            subprocess.run(["curl", "-L", "-o", video_out, target], capture_output=True)
    else:
        print("    Processing local file...")
        if not Path(target).exists():
            raise FileNotFoundError(f"File not found: {target}")
        import shutil
        shutil.copy(target, video_out)

    # Extract audio
    print("    Extracting audio...")
    subprocess.run([
        "ffmpeg", "-i", video_out, "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "1", audio_out, "-y"
    ], capture_output=True)

    return video_out, audio_out

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def get_face_cascade():
    """Find and load face cascade."""
    paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_default.xml"
    ]
    for p in paths:
        if Path(p).exists():
            return cv2.CascadeClassifier(p)
    return None

def calculate_laplacian_variance(roi) -> float:
    """Measure texture variance - low = suspiciously smooth."""
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_entropy(roi) -> float:
    """Calculate image entropy - synthetic content has abnormal entropy."""
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def check_boundary_consistency(frame, face_rect) -> Tuple[bool, float]:
    """Check for color inconsistency at face boundaries."""
    x, y, w, h = face_rect
    margin = 15

    # Get expanded region
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)

    face_roi = frame[y:y+h, x:x+w]
    boundary_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0 or boundary_roi.size == 0:
        return False, 0

    face_mean = np.mean(face_roi, axis=(0, 1))
    boundary_mean = np.mean(boundary_roi, axis=(0, 1))

    distance = np.sqrt(np.sum((face_mean - boundary_mean) ** 2))
    return distance > 40, distance

def analyze_visual(video_path: str) -> Tuple[float, float, float, List[dict]]:
    """
    Analyze video for visual manipulation.
    Returns: (manipulation_score, temporal_score, entropy_score, face_details)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    face_cascade = get_face_cascade()
    if face_cascade is None:
        print("    WARNING: Face cascade not found")
        return 0, 0, 0, []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, frame_count // 100)  # Sample ~100 frames

    suspicious_faces = 0
    total_faces = 0
    face_details = []

    # For temporal analysis
    prev_landmarks = None
    temporal_jumps = 0
    temporal_checks = 0

    # For entropy analysis
    entropy_values = []

    print(f"    Scanning {frame_count} frames (sampling every {sample_every})...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

        for (x, y, w, h) in faces:
            total_faces += 1
            face_roi = frame[y:y+h, x:x+w]

            # Texture analysis
            variance = calculate_laplacian_variance(face_roi)
            is_smooth = variance < 120

            # Boundary analysis
            boundary_issue, boundary_dist = check_boundary_consistency(frame, (x, y, w, h))

            # Entropy analysis
            entropy = calculate_entropy(face_roi)
            entropy_values.append(entropy)

            # Track suspicious faces
            suspicious = is_smooth or boundary_issue
            if suspicious:
                suspicious_faces += 1

            face_details.append({
                "frame": frame_idx,
                "variance": round(variance, 2),
                "entropy": round(entropy, 2),
                "boundary_distance": round(boundary_dist, 2),
                "suspicious": suspicious
            })

    cap.release()

    if total_faces == 0:
        print("    WARNING: No faces detected")
        return 0, 0, 0, []

    # Calculate scores
    visual_score = suspicious_faces / total_faces

    # Temporal score (placeholder - would need landmark tracking)
    temporal_score = 0.0

    # Entropy anomaly score
    if entropy_values:
        entropy_std = np.std(entropy_values)
        entropy_score = min(1.0, entropy_std / 2.0)  # High variance = suspicious
    else:
        entropy_score = 0.0

    print(f"    Analyzed {total_faces} faces, {suspicious_faces} suspicious ({visual_score*100:.1f}%)")

    return visual_score, temporal_score, entropy_score, face_details

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_audio(audio_path: str) -> float:
    """
    Analyze audio for synthesis artifacts.
    Synthetic voice often lacks high-frequency content.
    """
    try:
        with wave.open(audio_path, 'rb') as wav:
            n_frames = wav.getnframes()
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()

            # Read samples
            raw_data = wav.readframes(min(n_frames, sample_rate * 60))  # Max 60 seconds

            if n_channels == 1:
                samples = struct.unpack(f"{len(raw_data)//2}h", raw_data)
            else:
                # Take first channel
                all_samples = struct.unpack(f"{len(raw_data)//2}h", raw_data)
                samples = all_samples[::n_channels]

            samples = np.array(samples, dtype=np.float32) / 32768.0
    except Exception as e:
        print(f"    WARNING: Could not read audio: {e}")
        return 0.0

    if len(samples) < 2048:
        print("    WARNING: Audio too short for analysis")
        return 0.0

    # FFT analysis
    fft_size = 2048
    high_cutoff_bin = int(12000 * fft_size / sample_rate)

    anomalies = 0
    chunks = 0

    for i in range(0, len(samples) - fft_size, fft_size):
        chunk = samples[i:i + fft_size]
        spectrum = np.abs(np.fft.fft(chunk))[:fft_size // 2]

        total_energy = np.sum(spectrum ** 2)
        high_energy = np.sum(spectrum[high_cutoff_bin:] ** 2)

        if total_energy > 0:
            high_ratio = high_energy / total_energy
            if high_ratio < 0.05:  # Very little high-frequency content
                anomalies += 1

        chunks += 1
        if chunks >= 200:
            break

    if chunks == 0:
        return 0.0

    score = anomalies / chunks
    print(f"    Analyzed {chunks} audio chunks, {anomalies} anomalous ({score*100:.1f}%)")

    return score

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT ANALYSIS (OCR + SCAM DETECTION)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_context(video_path: str) -> Tuple[float, List[str]]:
    """
    Extract text from video and check for scam indicators.
    """
    import re

    scam_keywords = [
        "giveaway", "double", "2x", "return", "send eth", "send btc",
        "urgent", "limited time", "elon", "tesla", "free crypto", "claim now",
        "airdrop", "winner", "congratulations", "act now", "don't miss"
    ]

    wallet_pattern = re.compile(r"0x[a-fA-F0-9]{40}")
    btc_pattern = re.compile(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    full_text = ""
    flags = []
    num_samples = 12

    print(f"    Extracting text from {num_samples} sample frames...")

    for i in range(num_samples):
        pos = int(i * frame_count / num_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        ret, frame = cap.read()
        if not ret:
            continue

        # Save temp frame
        temp_path = f"temp_ocr_{i}.png"
        cv2.imwrite(temp_path, frame)

        # Run tesseract
        try:
            result = subprocess.run(
                ["tesseract", temp_path, "stdout", "-l", "eng", "--psm", "3"],
                capture_output=True, text=True, timeout=10
            )
            text = result.stdout.lower()
            full_text += text + " "
        except:
            pass

        # Cleanup
        if Path(temp_path).exists():
            os.remove(temp_path)

    cap.release()

    # Calculate score
    score = 0.0

    # Check for wallet addresses
    if wallet_pattern.search(full_text):
        flags.append("ETH wallet address detected")
        score += 0.5

    if btc_pattern.search(full_text):
        flags.append("BTC wallet address detected")
        score += 0.5

    # Check for scam keywords
    found_keywords = []
    for keyword in scam_keywords:
        if keyword in full_text:
            found_keywords.append(keyword)
            score += 0.1

    if found_keywords:
        flags.append(f"Scam keywords: {', '.join(found_keywords)}")

    return min(1.0, score), flags

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_annotated_video(video_path: str, report: VeritasReport, output_path: str = "veritas_output.mp4"):
    """Generate annotated video with overlays."""
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_cascade = get_face_cascade()
    metrics = report.metrics
    confidence = report.confidence

    print(f"[VERITAS-VIZ] Generating annotated video ({total_frames} frames)...")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Top HUD
        cv2.rectangle(frame, (0, 0), (width, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, 200), (0, 255, 255), 2)

        cv2.putText(frame, "PROJECT VERITAS // FORENSIC ANALYSIS",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Metrics
        def get_color(val, thresh=0.5):
            return (0, 0, 255) if val > thresh else (0, 255, 0)

        y_offset = 70
        cv2.putText(frame, f"Visual Manipulation: {metrics.visual_manipulation*100:.1f}%",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(metrics.visual_manipulation), 2)
        cv2.putText(frame, f"Audio Artifacts: {metrics.audio_artifacts*100:.1f}%",
                    (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(metrics.audio_artifacts), 2)
        cv2.putText(frame, f"Scam Indicators: {metrics.scam_indicators*100:.1f}%",
                    (20, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(metrics.scam_indicators, 0.3), 2)
        cv2.putText(frame, f"Entropy Anomaly: {metrics.entropy_anomaly*100:.1f}%",
                    (20, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, get_color(metrics.entropy_anomaly), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.1f}%",
                    (20, y_offset + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}",
                    (width - 180, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Face detection and analysis
        if face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                variance = calculate_laplacian_variance(face_roi)
                is_suspicious = variance < 120

                if is_suspicious:
                    color = (0, 0, 255)
                    label = f"SUSPICIOUS (Var:{variance:.0f})"
                    # Scan lines
                    for i in range(y, y+h, 6):
                        cv2.line(frame, (x, i), (x+w, i), (0, 0, 255), 1)
                else:
                    color = (0, 255, 0)
                    label = f"OK (Var:{variance:.0f})"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Bottom verdict
        cv2.rectangle(frame, (0, height-60), (width, height), (0, 0, 0), -1)

        if confidence > 0.75:
            verdict_color = (0, 0, 255)
            verdict_text = "VERDICT: HIGH PROBABILITY DEEPFAKE / SCAM"
        elif confidence > 0.5:
            verdict_color = (0, 165, 255)
            verdict_text = "VERDICT: POSSIBLE MANIPULATION DETECTED"
        else:
            verdict_color = (0, 255, 0)
            verdict_text = "VERDICT: LIKELY AUTHENTIC"

        cv2.putText(frame, verdict_text, (20, height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, verdict_color, 2)

        out.write(frame)

        if frame_num % 100 == 0:
            print(f"    Processed {frame_num}/{total_frames}...")

    cap.release()
    out.release()
    print(f"[VERITAS-VIZ] Saved to: {output_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(target: str, generate_video: bool = True) -> VeritasReport:
    """Run full VERITAS analysis."""
    print(f"\n{'='*60}")
    print("PROJECT VERITAS - Forensic Deepfake Detection")
    print(f"{'='*60}")
    print(f"Target: {target}\n")

    # 1. Ingest
    print("[*] Preparing video...")
    video_path, audio_path = download_video(target)

    # 2. Visual analysis
    print("\n[*] Analyzing visual content...")
    visual_score, temporal_score, entropy_score, face_details = analyze_visual(video_path)

    # 3. Audio analysis
    print("\n[*] Analyzing audio...")
    audio_score = analyze_audio(audio_path)

    # 4. Context analysis
    print("\n[*] Scanning for scam indicators...")
    context_score, flags = analyze_context(video_path)

    # 5. Calculate confidence
    confidence = (
        visual_score * 0.35 +
        audio_score * 0.20 +
        context_score * 0.25 +
        entropy_score * 0.20
    )

    if confidence > 0.75:
        verdict = "HIGH CONFIDENCE: SYNTHETIC/MANIPULATED CONTENT"
    elif confidence > 0.5:
        verdict = "MEDIUM CONFIDENCE: POSSIBLE MANIPULATION"
    else:
        verdict = "LOW CONFIDENCE: LIKELY AUTHENTIC"

    metrics = Metrics(
        visual_manipulation=visual_score,
        audio_artifacts=audio_score,
        scam_indicators=context_score,
        temporal_inconsistency=temporal_score,
        entropy_anomaly=entropy_score
    )

    report = VeritasReport(
        target=target,
        metrics=metrics,
        confidence=confidence,
        verdict=verdict,
        face_analysis=face_details[:20],  # Limit for JSON size
        flags=flags
    )

    # 6. Save report
    report_dict = asdict(report)
    # Convert numpy types to Python types for JSON
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    report_dict = convert_types(report_dict)
    with open("veritas_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[*] Report saved to: veritas_report.json")

    # 7. Generate annotated video
    if generate_video:
        print("\n[*] Generating annotated video...")
        generate_annotated_video(video_path, report)

    # 8. Cleanup
    for f in ["temp_video.mp4", "temp_audio.wav"]:
        if Path(f).exists():
            os.remove(f)

    # 9. Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Visual Manipulation: {visual_score*100:.1f}%")
    print(f"Audio Artifacts:     {audio_score*100:.1f}%")
    print(f"Scam Indicators:     {context_score*100:.1f}%")
    print(f"Entropy Anomaly:     {entropy_score*100:.1f}%")
    print(f"{'='*60}")
    print(f"CONFIDENCE: {confidence*100:.1f}%")
    print(f"VERDICT: {verdict}")
    if flags:
        print(f"\nFLAGS:")
        for flag in flags:
            print(f"  - {flag}")
    print(f"{'='*60}\n")

    return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python veritas.py <video_url_or_path> [--no-video]")
        print("\nExamples:")
        print("  python veritas.py https://youtube.com/watch?v=XXXXX")
        print("  python veritas.py /path/to/video.mp4")
        print("  python veritas.py video.mp4 --no-video")
        sys.exit(1)

    target = sys.argv[1]
    generate_video = "--no-video" not in sys.argv

    try:
        analyze(target, generate_video)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Make sure the file path is correct or the URL is valid.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[*] Analysis cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        print("Check that ffmpeg and tesseract are installed.")
        sys.exit(1)
