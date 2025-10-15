#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "════════════════════════════════════════════════════════════"
echo "🎤  Voice Transcriber - Installation Script"
echo "════════════════════════════════════════════════════════════"
echo

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}❌ Error: This tool is designed for macOS only${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Step 1: Checking system dependencies...${NC}"
echo

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}⚠️  Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo -e "${GREEN}✅ Homebrew is installed${NC}"
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  Python 3 not found. Installing Python...${NC}"
    brew install python3
else
    echo -e "${GREEN}✅ Python 3 is installed: $(python3 --version)${NC}"
fi

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}⚠️  ffmpeg not found. Installing ffmpeg...${NC}"
    brew install ffmpeg
else
    echo -e "${GREEN}✅ ffmpeg is installed${NC}"
fi

# Check for portaudio
if ! brew list portaudio &> /dev/null; then
    echo -e "${YELLOW}⚠️  portaudio not found. Installing portaudio...${NC}"
    brew install portaudio
else
    echo -e "${GREEN}✅ portaudio is installed${NC}"
fi

echo
echo -e "${BLUE}📋 Step 2: Setting up Python virtual environment...${NC}"
echo

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

echo
echo -e "${BLUE}📋 Step 3: Installing Python dependencies...${NC}"
echo

# Upgrade pip
pip3 install --upgrade pip --quiet

# Install requirements
echo "Installing Python packages (this may take a few minutes)..."
pip3 install -r requirements.txt

echo -e "${GREEN}✅ All Python dependencies installed${NC}"

echo
echo -e "${BLUE}📋 Step 4: Making scripts executable...${NC}"
echo

chmod +x transcribe.py
chmod +x run.sh
chmod +x detect_key.py

echo -e "${GREEN}✅ Scripts are now executable${NC}"

echo
echo -e "${BLUE}📋 Step 5: Creating command alias (optional)...${NC}"
echo

# Detect shell
SHELL_CONFIG=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
fi

if [ -n "$SHELL_CONFIG" ]; then
    ALIAS_LINE="alias transcribe='cd \"$SCRIPT_DIR\" && ./run.sh'"

    if grep -q "alias transcribe=" "$SHELL_CONFIG" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  'transcribe' alias already exists in $SHELL_CONFIG${NC}"
    else
        echo
        read -p "Would you like to add 'transcribe' command to your shell? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "" >> "$SHELL_CONFIG"
            echo "# Voice Transcriber" >> "$SHELL_CONFIG"
            echo "$ALIAS_LINE" >> "$SHELL_CONFIG"
            echo -e "${GREEN}✅ Added 'transcribe' alias to $SHELL_CONFIG${NC}"
            echo -e "${YELLOW}   Run 'source $SHELL_CONFIG' or restart your terminal${NC}"
        else
            echo -e "${YELLOW}⏭️  Skipped alias creation${NC}"
        fi
    fi
fi

echo
echo "════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Installation complete!${NC}"
echo "════════════════════════════════════════════════════════════"
echo

echo -e "${BLUE}🚀 How to run:${NC}"
echo
echo "  Option 1: Run directly"
echo "    cd $SCRIPT_DIR"
echo "    ./run.sh"
echo
if grep -q "alias transcribe=" "$SHELL_CONFIG" 2>/dev/null; then
    echo "  Option 2: Use the 'transcribe' command"
    echo "    transcribe"
    echo
fi

echo -e "${BLUE}⚙️  Configuration:${NC}"
echo "  Edit config.yaml to customize settings"
echo "  Run ./detect_key.py to find your hotkey code"
echo

echo -e "${BLUE}🔒 Permissions needed:${NC}"
echo "  • Microphone access (System Preferences → Privacy → Microphone)"
echo "  • Accessibility access for auto-paste (System Preferences → Privacy → Accessibility)"
echo

echo -e "${GREEN}Happy transcribing! 🎙️${NC}"
echo
