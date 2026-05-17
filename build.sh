#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════"
echo "🔨  Voice Transcriber - Build Standalone Binary"
echo "════════════════════════════════════════════════════════════"
echo

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "   Run ./install.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo -e "${BLUE}📋 Step 1: Installing PyInstaller...${NC}"
echo

pip install pyinstaller --quiet
echo -e "${GREEN}✅ PyInstaller installed${NC}"

echo
echo -e "${BLUE}📋 Step 2: Building standalone binary...${NC}"
echo -e "${YELLOW}   (This may take several minutes)${NC}"
echo

# Build with PyInstaller
pyinstaller --clean \
    --onefile \
    --name "VoiceTranscriber" \
    --add-data "config.yaml:." \
    --hidden-import=parakeet_mlx \
    --hidden-import=mlx \
    --hidden-import=pyaudio \
    --hidden-import=pynput \
    --hidden-import=pyperclip \
    --hidden-import=yaml \
    --hidden-import=numpy \
    --hidden-import=soundfile \
    --collect-all parakeet_mlx \
    --collect-all mlx \
    transcribe.py

echo
echo -e "${GREEN}✅ Build complete!${NC}"
echo

echo "════════════════════════════════════════════════════════════"
echo -e "${GREEN}📦 Binary created successfully!${NC}"
echo "════════════════════════════════════════════════════════════"
echo

echo -e "${BLUE}📍 Location:${NC}"
echo "   $SCRIPT_DIR/dist/VoiceTranscriber"
echo

echo -e "${BLUE}🚀 To run the binary:${NC}"
echo "   ./dist/VoiceTranscriber"
echo

echo -e "${BLUE}📤 To distribute:${NC}"
echo "   1. Copy dist/VoiceTranscriber to any Mac"
echo "   2. Copy config.yaml alongside it (optional)"
echo "   3. Run it!"
echo

echo -e "${YELLOW}⚠️  Note:${NC}"
echo "   The binary bundles parakeet-mlx + MLX runtime (Apple Silicon only)"
echo "   First run will still download the Parakeet model (~600MB)"
echo

echo -e "${GREEN}Done! 🎉${NC}"
echo
