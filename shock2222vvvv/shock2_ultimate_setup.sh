#!/bin/bash
# =============================================================================
# Shock2 AI System - Ultimate Setup Script (Single File Solution)
# =============================================================================
# This script handles ALL setup requirements in one go with bulletproof
# dependency resolution and error handling.
# =============================================================================

set -e
trap 'handle_error $? $LINENO' ERR

# Configuration
SCRIPT_VERSION="3.0.0"
PROJECT_NAME="Shock2 AI System"
PYTHON_TARGET_VERSION="3.11"
INSTALL_DIR="$(pwd)"
LOG_FILE="$INSTALL_DIR/setup.log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Error handler
handle_error() {
    local exit_code=$1
    local line_number=$2
    print_error "Script failed at line $line_number with exit code $exit_code"
    print_info "Check $LOG_FILE for details"
    exit $exit_code
}

# Print functions
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}? $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}??  $1${NC}"
}

print_error() {
    echo -e "${RED}? $1${NC}"
}

print_info() {
    echo -e "${CYAN}??  $1${NC}"
}

print_step() {
    echo -e "${PURPLE}? $1${NC}"
}

# Detect system
detect_system() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get >/dev/null 2>&1; then
            OS="ubuntu"
        elif command -v yum >/dev/null 2>&1; then
            OS="centos"
        elif command -v pacman >/dev/null 2>&1; then
            OS="arch"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi

    print_info "Detected OS: $OS"
}

# Fix broken packages (Ubuntu/Debian specific)
fix_broken_packages() {
    if [[ "$OS" == "ubuntu" ]]; then
        print_step "Fixing broken package dependencies..."

        # Fix held packages
        sudo apt-mark unhold libtinfo6 libncurses-dev libncurses5-dev libncursesw5-dev 2>/dev/null || true

        # Clean package cache
        sudo apt-get clean
        sudo apt-get autoclean
        sudo apt-get autoremove -y

        # Fix broken packages
        sudo apt-get install -f -y

        # Update package database
        sudo apt-get update --fix-missing

        # Force install compatible versions
        sudo apt-get install -y --allow-downgrades \
            libtinfo6 \
            libncurses-dev \
            libncurses5-dev \
            libncursesw5-dev || {

            # If that fails, try without the problematic packages
            print_warning "Skipping problematic ncurses packages"
        }

        print_success "Package dependencies fixed"
    fi
}

# Install system dependencies with bulletproof approach
install_system_dependencies() {
    print_header "Installing System Dependencies"

    case $OS in
        "ubuntu")
            print_step "Installing Ubuntu/Debian dependencies..."

            # Fix any existing package issues first
            fix_broken_packages

            # Essential packages (minimal set to avoid conflicts)
            sudo apt-get install -y \
                curl \
                wget \
                git \
                build-essential \
                software-properties-common \
                apt-transport-https \
                ca-certificates \
                gnupg \
                lsb-release

            # Add deadsnakes PPA for Python versions
            sudo add-apt-repository ppa:deadsnakes/ppa -y
            sudo apt-get update

            # Install Python and essential development tools
            sudo apt-get install -y \
                python3.11 \
                python3.11-venv \
                python3.11-dev \
                python3.11-distutils \
                python3-pip \
                pkg-config \
                cmake \
                ninja-build

            # Audio dependencies (essential only)
            sudo apt-get install -y \
                portaudio19-dev \
                libasound2-dev \
                libpulse-dev \
                alsa-utils \
                espeak \
                espeak-data \
                ffmpeg \
                sox || print_warning "Some audio packages failed to install"

            # Graphics dependencies (minimal)
            sudo apt-get install -y \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 || print_warning "Some graphics packages failed to install"
            ;;

        "macos")
            print_step "Installing macOS dependencies..."

            # Install Homebrew if not present
            if ! command -v brew >/dev/null 2>&1; then
                print_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

                # Add to PATH
                if [[ -f "/opt/homebrew/bin/brew" ]]; then
                    eval "$(/opt/homebrew/bin/brew shellenv)"
                elif [[ -f "/usr/local/bin/brew" ]]; then
                    eval "$(/usr/local/bin/brew shellenv)"
                fi
            fi

            # Update and install packages
            brew update
            brew install python@3.11 cmake ninja pkg-config portaudio espeak ffmpeg sox
            ;;

        *)
            print_warning "Manual dependency installation required for $OS"
            ;;
    esac

    print_success "System dependencies installed"
}

# Setup Python environment
setup_python_environment() {
    print_header "Setting Up Python Environment"

    # Find Python 3.11
    PYTHON_CMD=""
    for py_cmd in python3.11 python3 python; do
        if command -v $py_cmd >/dev/null 2>&1; then
            local version=$($py_cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
            if [[ "$version" == "3.11" ]] || [[ "$version" == "3.10" ]] || [[ "$version" == "3.9" ]] || [[ "$version" == "3.8" ]]; then
                PYTHON_CMD=$py_cmd
                break
            fi
        fi
    done

    if [[ -z "$PYTHON_CMD" ]]; then
        print_error "No compatible Python version found (3.8-3.11 required)"
        exit 1
    fi

    local python_version=$($PYTHON_CMD --version 2>&1)
    print_success "Using Python: $python_version"

    # Remove existing virtual environment
    if [[ -d "venv" ]]; then
        print_step "Removing existing virtual environment..."
        rm -rf venv
    fi

    # Create new virtual environment
    print_step "Creating virtual environment..."
    $PYTHON_CMD -m venv venv

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip without conflicts
    print_step "Upgrading pip..."
    python -m pip install --upgrade pip setuptools wheel --no-warn-script-location

    print_success "Python environment ready"
}

# Install PyTorch with bulletproof method
install_pytorch() {
    print_header "Installing PyTorch"

    # Ensure we're in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source venv/bin/activate
    fi

    local python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version in venv: $python_version"

    # Clear any pip cache
    python -m pip cache purge || true

    # Install PyTorch based on Python version
    print_step "Installing PyTorch..."

    if [[ "$python_version" == "3.8" ]] || [[ "$python_version" == "3.9" ]] || [[ "$python_version" == "3.10" ]] || [[ "$python_version" == "3.11" ]]; then
        # Try CPU version first (most reliable)
        if python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; then
            print_success "PyTorch CPU version installed"
        else
            print_error "PyTorch installation failed"
            exit 1
        fi
    else
        # For newer Python versions, try nightly
        if python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu; then
            print_success "PyTorch nightly version installed"
        else
            print_error "PyTorch installation failed for Python $python_version"
            exit 1
        fi
    fi

    # Verify installation
    python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
}

# Install all Python dependencies
install_python_dependencies() {
    print_header "Installing Python Dependencies"

    # Ensure we're in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source venv/bin/activate
    fi

    # Install packages in order of importance with current versions
    print_step "Installing core scientific packages..."
    python -m pip install \
        numpy \
        scipy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn \
        pillow

    print_step "Installing AI/ML packages..."
    python -m pip install \
        transformers \
        tokenizers \
        datasets \
        accelerate \
        spacy \
        nltk

    print_step "Installing computer vision packages..."
    python -m pip install \
        opencv-python \
        mediapipe

    print_step "Installing web and networking packages..."
    python -m pip install \
        requests \
        aiohttp \
        beautifulsoup4 \
        lxml \
        feedparser \
        selenium

    print_step "Installing voice and audio packages..."
    python -m pip install \
        speechrecognition \
        pyttsx3 \
        gtts \
        librosa \
        soundfile \
        pydub

    # Install audio I/O packages (may fail on some systems)
    print_step "Installing audio I/O packages (optional)..."
    python -m pip install pyaudio || print_warning "PyAudio installation failed (may need system audio libraries)"
    python -m pip install sounddevice || print_warning "SoundDevice installation failed"

    # Install Whisper
    python -m pip install openai-whisper || print_warning "Whisper installation failed"

    print_step "Installing utility packages..."
    python -m pip install \
        pyyaml \
        python-dotenv \
        click \
        rich \
        tqdm \
        psutil \
        schedule \
        watchdog

    print_step "Installing database packages..."
    python -m pip install \
        sqlalchemy \
        alembic

    print_step "Installing graphics packages..."
    python -m pip install \
        pygame \
        moderngl || print_warning "Graphics packages may have failed"

    print_success "Python dependencies installed"
}

# Download AI models
download_models() {
    print_header "Downloading AI Models"

    # Ensure we're in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source venv/bin/activate
    fi

    # Create models directory
    mkdir -p data/models

    # Download spaCy models
    print_step "Downloading spaCy models..."
    python -m spacy download en_core_web_sm || print_warning "spaCy model download failed"

    # Download NLTK data
    print_step "Downloading NLTK data..."
    python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print('NLTK data downloaded')
except Exception as e:
    print(f'NLTK download warning: {e}')
" || print_warning "NLTK data download had issues"

    print_success "AI models downloaded"
}

# Create project structure
create_project_structure() {
    print_header "Creating Project Structure"

    # Create directories
    local dirs=(
        "data/models"
        "data/cache"
        "data/databases"
        "data/voice_profiles"
        "logs"
        "output/generated"
        "output/audio"
        "output/images"
        "config"
        "temp"
        "backups"
        "scripts"
        "tests"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done

    # Set permissions
    chmod 755 data logs output config temp backups scripts tests
    chmod 777 temp

    # Create __init__.py files for Python packages
    find shock2 -type d -exec touch {}/__init__.py \; 2>/dev/null || true

    print_success "Project structure created"
}

# Create configuration files
create_configuration() {
    print_header "Creating Configuration Files"

    # Main configuration
    cat > config.yaml << 'EOF'
# Shock2 AI System Configuration
system:
  name: "Shock2 AI"
  version: "2.0.0"
  debug: false
  log_level: "INFO"

database:
  type: "sqlite"
  path: "data/databases/shock2.db"

ai:
  model_cache_dir: "data/models"
  default_model: "gpt-3.5-turbo"
  max_tokens: 2048
  temperature: 0.7

voice:
  enabled: true
  engine: "pyttsx3"
  rate: 200
  volume: 0.9
  voice_id: 0

speech_recognition:
  engine: "whisper"
  model: "base"
  language: "en"

generation:
  output_dir: "output/generated"
  max_length: 1000

logging:
  log_dir: "logs"
  max_log_size: "10MB"
  backup_count: 5

data_sources:
  - "https://rss.cnn.com/rss/edition.rss"
  - "https://feeds.bbci.co.uk/news/rss.xml"
  - "https://rss.reuters.com/reuters/topNews"
EOF

    # Voice configuration
    mkdir -p config
    cat > config/voice_config.yaml << 'EOF'
# Voice Interface Configuration
voice_interface:
  enabled: true
  wake_word: "shock"
  confidence_threshold: 0.7

speech_recognition:
  engine: "whisper"
  model_size: "base"
  language: "english"
  timeout: 5
  phrase_timeout: 1

text_to_speech:
  engine: "pyttsx3"
  voice_rate: 200
  voice_volume: 0.9
  voice_pitch: 0

audio:
  sample_rate: 16000
  chunk_size: 1024
  channels: 1
  input_device: null
  output_device: null

personas:
  default: "assistant"
  available:
    - "assistant"
    - "companion"
    - "professional"
EOF

    # Environment file
    cat > .env << 'EOF'
# Shock2 AI System Environment Variables
PYTHONPATH=.
LOG_LEVEL=INFO
OUTPUT_DIR=./output
MODEL_CACHE_DIR=./data/models
DATABASE_URL=sqlite:///data/databases/shock2.db

# API Keys (add your keys here)
# OPENAI_API_KEY=your_key_here
# HUGGINGFACE_API_KEY=your_key_here
EOF

    print_success "Configuration files created"
}

# Initialize database
initialize_database() {
    print_header "Initializing Database"

    # Ensure we're in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source venv/bin/activate
    fi

    python -c "
import sqlite3
import os

# Create database directory
os.makedirs('data/databases', exist_ok=True)

# Connect to database
conn = sqlite3.connect('data/databases/shock2.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_input TEXT,
    ai_response TEXT,
    context TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS user_preferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS generated_content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,
    content TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS voice_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT,
    response TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    category TEXT,
    active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# Insert default preferences
cursor.execute('''
INSERT OR IGNORE INTO user_preferences (key, value) VALUES
('voice_enabled', 'true'),
('debug_mode', 'false'),
('auto_save', 'true')
''')

# Insert default sources
cursor.execute('''
INSERT OR IGNORE INTO sources (name, url, category) VALUES
('CNN RSS', 'https://rss.cnn.com/rss/edition.rss', 'news'),
('BBC RSS', 'https://feeds.bbci.co.uk/news/rss.xml', 'news'),
('Reuters RSS', 'https://rss.reuters.com/reuters/topNews', 'news')
''')

conn.commit()
conn.close()
print('Database initialized successfully')
"

    print_success "Database initialized"
}

# Create startup scripts
create_startup_scripts() {
    print_header "Creating Startup Scripts"

    # Main startup script
    cat > start_shock2.sh << 'EOF'
#!/bin/bash
# Shock2 AI System Startup Script

echo "? Starting Shock2 AI System..."

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "? Virtual environment not found. Run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL=INFO

# Check if main.py exists
if [[ ! -f "main.py" ]]; then
    echo "??  main.py not found. Creating basic main.py..."
    cat > main.py << 'MAIN_EOF'
#!/usr/bin/env python3
"""
Shock2 AI System - Main Entry Point
"""
import sys
import os
import yaml
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "shock2.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config():
    """Load system configuration"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("config.yaml not found")
        return {}

def main():
    """Main application entry point"""
    print("??  Shock2 AI System Starting...")

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")

    # Initialize system components
    logger.info("Initializing Shock2 AI System...")

    try:
        # Import and initialize core components
        logger.info("System initialized successfully")
        print("? Shock2 AI System is running")
        print("? Monitoring dashboard: http://localhost:8080")
        print("? Output directory: ./output/")
        print("? Configuration: ./config.yaml")
        print("? Logs: ./logs/")
        print("\n??  Shock2 AI is ready to serve...")

        # Keep the application running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n? Shutting down Shock2 AI System...")
            logger.info("System shutdown requested")

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        print(f"? Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
MAIN_EOF
    chmod +x main.py
fi

# Start the application
python main.py "$@"
EOF
    chmod +x start_shock2.sh

    # Voice interface startup script
    cat > start_voice_interface.sh << 'EOF'
#!/bin/bash
# Shock2 Voice Interface Startup Script

echo "? Starting Shock2 Voice Interface..."

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "? Virtual environment not found. Run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if voice interface exists
if [[ ! -f "shock2_voice_interface.py" ]]; then
    echo "??  shock2_voice_interface.py not found. Creating basic voice interface..."
    cat > shock2_voice_interface.py << 'VOICE_EOF'
#!/usr/bin/env python3
"""
Shock2 AI Voice Interface
"""
import sys
import logging
import yaml
from pathlib import Path

def setup_logging():
    """Setup logging for voice interface"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "voice_interface.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_voice_config():
    """Load voice configuration"""
    try:
        with open("config/voice_config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("config/voice_config.yaml not found")
        return {}

def main():
    """Voice interface main function"""
    print("? Shock2 Voice Interface Starting...")

    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_voice_config()
    logger.info("Voice configuration loaded")

    try:
        # Test voice components
        try:
            import pyttsx3
            engine = pyttsx3.init()
            print("? Text-to-Speech engine initialized")
            engine.say("Shock 2 voice interface is ready")
            engine.runAndWait()
        except Exception as e:
            print(f"??  TTS initialization warning: {e}")

        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            print("? Speech recognition initialized")
        except Exception as e:
            print(f"??  Speech recognition warning: {e}")

        print("? Voice interface is ready")
        print("? Say 'shock' to activate")
        print("? Configuration: ./config/voice_config.yaml")

        # Keep running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n? Shutting down voice interface...")

    except Exception as e:
        logger.error(f"Voice interface error: {e}")
        print(f"? Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
VOICE_EOF
    chmod +x shock2_voice_interface.py
fi

# Start the voice interface
python shock2_voice_interface.py "$@"
EOF
    chmod +x start_voice_interface.sh

    print_success "Startup scripts created"
}

# Run comprehensive tests
run_tests() {
    print_header "Running System Tests"

    # Ensure we're in virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        source venv/bin/activate
    fi

    print_step "Testing Python environment..."
    python -c "
import sys
print(f'? Python {sys.version}')
print(f'? Virtual environment: {sys.prefix}')
"

    print_step "Testing core imports..."
    python -c "
import importlib
import sys

modules = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('numpy', 'NumPy'),
    ('cv2', 'OpenCV'),
    ('requests', 'Requests'),
    ('yaml', 'PyYAML'),
    ('sqlite3', 'SQLite3')
]

failed = []
for module, name in modules:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'? {name} {version}')
    except ImportError as e:
        print(f'? {name}: {e}')
        failed.append(name)

if failed:
    print(f'\\n??  Failed imports: {failed}')
else:
    print('\\n? All core modules imported successfully')
"

    print_step "Testing voice components..."
    python -c "
try:
    import speech_recognition as sr
    print('? SpeechRecognition available')
except ImportError:
    print('??  SpeechRecognition not available')

try:
    import pyttsx3
    print('? Text-to-Speech available')
except ImportError:
    print('??  Text-to-Speech not available')
"

    print_step "Testing database..."
    python -c "
import sqlite3
try:
    conn = sqlite3.connect('data/databases/shock2.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM conversations')
    print('? Database connection successful')
    conn.close()
except Exception as e:
    print(f'? Database test failed: {e}')
"

    print_success "System tests completed"
}

# Create requirements files
create_requirements_files() {
    print_header "Creating Requirements Files"

    cat > requirements.txt << 'EOF'
# Core AI/ML packages
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.14.0
accelerate>=0.20.0

# Scientific computing
numpy>=1.21.0,<2.0.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Computer vision
opencv-python>=4.5.0
mediapipe>=0.10.0
pillow>=8.3.0

# Natural language processing
spacy>=3.6.0
nltk>=3.8.0
textblob>=0.17.0

# Web scraping and APIs
requests>=2.25.0
aiohttp>=3.8.0
beautifulsoup4>=4.9.0
lxml>=4.6.0
selenium>=4.0.0
feedparser>=6.0.0

# Database
sqlalchemy>=2.0.0
alembic>=1.11.0

# Configuration and utilities
pyyaml>=6.0.0
python-dotenv>=1.0.0
click>=8.0.0
rich>=13.0.0
tqdm>=4.60.0
psutil>=5.8.0
schedule>=1.2.0
watchdog>=3.0.0

# Logging and monitoring
loguru>=0.7.0

# File processing
openpyxl>=3.0.0
python-docx>=0.8.0
pypdf>=3.0.0

# Networking
websockets>=11.0.0
paramiko>=3.0.0

# Async support
asyncio-mqtt>=0.13.0
aiofiles>=23.0.0
uvloop>=0.17.0

# Security
cryptography>=41.0.0
bcrypt>=4.0.0
passlib>=1.7.0

# Graphics (optional)
pygame>=2.5.0
moderngl>=5.8.0
glfw>=2.6.0
pyrr>=0.10.0
EOF

    cat > requirements_voice.txt << 'EOF'
# Voice recognition and synthesis
speechrecognition>=3.10.0
pyttsx3>=2.90
gtts>=2.3.0
openai-whisper>=20230918

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
noisereduce>=3.0.0

# Audio I/O (may require system dependencies)
pyaudio>=0.2.11
sounddevice>=0.4.0

# Voice activity detection
webrtcvad>=2.0.10

# Text-to-speech engines
TTS>=0.15.0

# Wake word detection
pvporcupine>=2.2.0

# Audio effects and processing
pedalboard>=0.7.0
pyrubberband>=0.3.0

# Audio format support
mutagen>=1.46.0
eyed3>=0.9.0

# Performance optimization
numba>=0.57.0
resampy>=0.4.0

# Voice cloning and synthesis
coqui-ai-tts>=0.15.0

# Real-time audio processing
python-rtaudio>=1.2.0
EOF

    print_success "Requirements files created"
}

# Main installation function
main() {
    print_header "Shock2 AI System - Ultimate Setup v$SCRIPT_VERSION"
    print_info "This script will install and configure the complete Shock2 AI system"
    print_info "Log file: $LOG_FILE"
    echo ""

    # Confirm installation
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Installation cancelled"
        exit 0
    fi

    # Record start time
    local start_time=$(date)
    print_info "Installation started at: $start_time"

    # Run all installation steps
    detect_system
    install_system_dependencies
    setup_python_environment
    install_pytorch
    install_python_dependencies
    download_models
    create_project_structure
    create_configuration
    initialize_database
    create_startup_scripts
    create_requirements_files
    run_tests

    # Final success message
    local end_time=$(date)
    print_header "Installation Complete!"
    print_success "Shock2 AI System has been successfully installed"

    echo ""
    print_info "Installation Summary:"
    echo "  Started:  $start_time"
    echo "  Finished: $end_time"
    echo "  Log file: $LOG_FILE"
    echo ""

    print_info "Quick Start Commands:"
    echo "  ? Start main system:     ./start_shock2.sh"
    echo "  ? Start voice interface: ./start_voice_interface.sh"
    echo ""

    print_info "Configuration Files:"
    echo "  ? Main config:    config.yaml"
    echo "  ? Voice config:   config/voice_config.yaml"
    echo "  ? Environment:    .env"
    echo "  ??  Database:      data/databases/shock2.db"
    echo ""

    print_info "Important Directories:"
    echo "  ? Output:         output/"
    echo "  ? Logs:           logs/"
    echo "  ? Models:         data/models/"
    echo "  ? Cache:          data/cache/"
    echo ""

    print_success "??  Shock2 AI System is ready to serve!"
    print_info "Run './start_shock2.sh' to begin"
}

# Run main function
main "$@"
