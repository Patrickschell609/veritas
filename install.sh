#!/bin/bash
#
# VERITAS Installer - Bulletproof setup for forensic deepfake detection
# Supports: Linux (Debian/Ubuntu/Kali), macOS
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           PROJECT VERITAS - INSTALLER                     ║${NC}"
echo -e "${CYAN}║           Forensic Deepfake Detection                     ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ -f /etc/debian_version ]]; then
        echo "debian"
    elif [[ -f /etc/redhat-release ]]; then
        echo "redhat"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo -e "[*] Detected OS: ${CYAN}$OS${NC}"

# Check for root on Linux
if [[ "$OS" != "macos" ]] && [[ "$EUID" -ne 0 ]]; then
    SUDO="sudo"
else
    SUDO=""
fi

# Track what we install
MISSING_DEPS=()

# ─────────────────────────────────────────────────────────────────────────────
# CHECK FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

check_python() {
    echo -n "[*] Checking Python 3... "
    if command -v python3 &> /dev/null; then
        PY_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
        echo -e "${GREEN}OK${NC} (Python $PY_VERSION)"
        return 0
    else
        echo -e "${RED}NOT FOUND${NC}"
        MISSING_DEPS+=("python3")
        return 1
    fi
}

check_ffmpeg() {
    echo -n "[*] Checking ffmpeg... "
    if command -v ffmpeg &> /dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}NOT FOUND${NC}"
        MISSING_DEPS+=("ffmpeg")
        return 1
    fi
}

check_tesseract() {
    echo -n "[*] Checking tesseract-ocr... "
    if command -v tesseract &> /dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${YELLOW}NOT FOUND${NC} (optional - OCR disabled)"
        return 0  # Not critical
    fi
}

check_opencv() {
    echo -n "[*] Checking OpenCV... "
    if python3 -c "import cv2" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${YELLOW}WILL INSTALL${NC}"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# INSTALL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

install_macos() {
    echo ""
    echo -e "[*] Installing dependencies for ${CYAN}macOS${NC}..."

    # Check Homebrew
    if ! command -v brew &> /dev/null; then
        echo -e "${YELLOW}[!] Homebrew not found. Installing...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    echo "[*] Installing system packages..."
    brew install python3 ffmpeg tesseract 2>/dev/null || true

    echo -e "${GREEN}[OK]${NC} System dependencies installed"
}

install_debian() {
    echo ""
    echo -e "[*] Installing dependencies for ${CYAN}Debian/Ubuntu${NC}..."

    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq python3 python3-venv python3-pip ffmpeg tesseract-ocr libgl1-mesa-glx

    echo -e "${GREEN}[OK]${NC} System dependencies installed"
}

install_redhat() {
    echo ""
    echo -e "[*] Installing dependencies for ${CYAN}RHEL/Fedora${NC}..."

    $SUDO dnf install -y python3 python3-pip ffmpeg tesseract

    echo -e "${GREEN}[OK]${NC} System dependencies installed"
}

setup_venv() {
    echo ""
    echo "[*] Setting up Python virtual environment..."

    if [ -d ".venv" ]; then
        echo "    Removing old environment..."
        rm -rf .venv
    fi

    python3 -m venv .venv

    echo "[*] Installing Python packages..."
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -q \
        opencv-python \
        numpy \
        yt-dlp \
        pillow \
        mediapipe \
        librosa \
        scipy

    echo -e "${GREEN}[OK]${NC} Python environment ready"
}

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

verify_install() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "                    VERIFICATION"
    echo "═══════════════════════════════════════════════════════════════"

    FAILED=0

    # Check Python packages
    echo -n "[*] opencv-python... "
    if .venv/bin/python -c "import cv2; print(cv2.__version__)" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        FAILED=1
    fi

    echo -n "[*] numpy... "
    if .venv/bin/python -c "import numpy; print(numpy.__version__)" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        FAILED=1
    fi

    echo -n "[*] yt-dlp... "
    if .venv/bin/python -c "import yt_dlp" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        FAILED=1
    fi

    echo -n "[*] ffmpeg... "
    if command -v ffmpeg &> /dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        FAILED=1
    fi

    # Test import of main script
    echo -n "[*] veritas.py... "
    if .venv/bin/python -c "import sys; sys.path.insert(0, '.'); from veritas import Metrics" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        FAILED=1
    fi

    return $FAILED
}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    DEPENDENCY CHECK"
echo "═══════════════════════════════════════════════════════════════"

check_python
check_ffmpeg
check_tesseract

# Install if needed
if [ ${#MISSING_DEPS[@]} -gt 0 ] || ! check_opencv; then
    echo ""
    echo -e "${YELLOW}[!] Missing dependencies detected. Installing...${NC}"

    case $OS in
        macos)
            install_macos
            ;;
        debian)
            install_debian
            ;;
        redhat)
            install_redhat
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown OS. Please install manually:${NC}"
            echo "  - python3"
            echo "  - ffmpeg"
            echo "  - tesseract (optional)"
            exit 1
            ;;
    esac
fi

# Always setup/refresh venv
setup_venv

# Verify
if verify_install; then
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                INSTALLATION COMPLETE!                         ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Usage:"
    echo ""
    echo -e "    ${CYAN}./VERITAS.sh${NC}                         # Launch GUI"
    echo -e "    ${CYAN}./run.sh <url or file>${NC}               # Command line"
    echo ""
    echo "  Examples:"
    echo ""
    echo "    ./run.sh https://youtube.com/watch?v=ABC123"
    echo "    ./run.sh suspicious_video.mp4"
    echo ""

    # Make scripts executable
    chmod +x run.sh VERITAS.sh 2>/dev/null || true

else
    echo ""
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}                INSTALLATION FAILED                            ${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Some components failed to install. Please check errors above."
    exit 1
fi
