#!/bin/bash
# Shock2 Voice Interface Setup Script

set -e

echo "🚀 Setting up Shock2 Voice Interface..."
echo "======================================"

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || {
    echo "❌ Python 3.8+ is required"
    exit 1
}

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install system dependencies based on OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Installing Linux dependencies..."
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio espeak espeak-data libespeak1 libespeak-dev flac ffmpeg libsndfile1-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Installing macOS dependencies..."
    if command -v brew &> /dev/null; then
        brew install portaudio espeak flac ffmpeg
    else
        echo "⚠️ Homebrew not found. Please install manually:"
        echo "   - PortAudio"
        echo "   - eSpeak"
        echo "   - FLAC"
        echo "   - FFmpeg"
    fi
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "🪟 Windows detected. Please install manually:"
    echo "   - Microsoft Visual C++ Build Tools"
    echo "   - PyAudio wheel"
    echo "   - eSpeak"
fi

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements_voice.txt

# Download spaCy models
echo "🧠 Downloading spaCy models..."
python -m spacy download en_core_web_sm || echo "⚠️ Failed to download en_core_web_sm"
python -m spacy download en_core_web_md || echo "⚠️ Failed to download en_core_web_md"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p assets data/models logs output

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.sh
chmod +x shock2_voice_interface.py

# Run installation verification
echo "🔍 Verifying installation..."
python install_voice_dependencies.py

echo ""
echo "✅ Shock2 Voice Interface setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Ensure your microphone is connected and working"
echo "   2. Test audio output (speakers/headphones)"
echo "   3. Run the voice interface: python shock2_voice_interface.py"
echo ""
echo "🎤 Voice Commands Examples:"
echo "   - 'Generate news about artificial intelligence'"
echo "   - 'Show me system status'"
echo "   - 'Activate stealth mode'"
echo "   - 'Analyze performance metrics'"
echo "   - 'Make an autonomous decision'"
echo ""
echo "🕶️ Shock2 AI is ready to serve..."
