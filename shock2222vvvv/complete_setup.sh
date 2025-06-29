#!/bin/bash
# =============================================================================
# Shock2 AI System - Complete Setup Script
# =============================================================================
# This script automates the complete installation and configuration of the
# Shock2 AI system, including all dependencies, models, and configurations.
# =============================================================================

set -e

# Default configuration
PYTHON_CMD="${PYTHON_CMD:-python3}"
INSTALL_GPU=true
INSTALL_VOICE=true
INSTALL_DEV=false
MINIMAL_INSTALL=false
DOCKER_SETUP=false
FORCE_REINSTALL=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Utility functions
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

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-gpu)
                INSTALL_GPU=false
                shift
                ;;
            --voice-only)
                INSTALL_VOICE=true
                INSTALL_GPU=false
                MINIMAL_INSTALL=true
                shift
                ;;
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            --minimal)
                MINIMAL_INSTALL=true
                shift
                ;;
            --docker)
                DOCKER_SETUP=true
                shift
                ;;
            --force)
                FORCE_REINSTALL=true
                shift
                ;;
            --python)
                PYTHON_CMD="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "Shock2 AI System Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-gpu        Skip GPU/CUDA installation"
    echo "  --voice-only    Install only voice interface components"
    echo "  --dev           Install development dependencies"
    echo "  --minimal       Minimal installation (core components only)"
    echo "  --docker        Set up Docker environment"
    echo "  --force         Force reinstallation of all components"
    echo "  --python CMD    Specify Python command (default: python3)"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Full installation"
    echo "  $0 --no-gpu                     # CPU-only installation"
    echo "  $0 --voice-only                 # Voice interface only"
    echo "  $0 --python python3.11          # Use specific Python version"
}

# Check system requirements
check_system_requirements() {
    print_header "Checking System Requirements"

    # Check Python version
    if ! command -v $PYTHON_CMD >/dev/null 2>&1; then
        print_error "Python command '$PYTHON_CMD' not found"
        print_info "Try installing compatible Python first:"
        print_info "  ./install_compatible_python.sh"
        exit 1
    fi

    local python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    local major_version=$(echo $python_version | cut -d'.' -f1)
    local minor_version=$(echo $python_version | cut -d'.' -f2)

    print_info "Found Python $python_version"

    # Check if Python version is compatible
    if [[ $major_version -eq 3 ]] && [[ $minor_version -ge 8 ]] && [[ $minor_version -le 12 ]]; then
        print_success "Python version is compatible"
    elif [[ $major_version -eq 3 ]] && [[ $minor_version -ge 13 ]]; then
        print_warning "Python $python_version is very new - some packages may not be available"
        print_info "Consider using Python 3.11 for best compatibility"
    else
        print_error "Python version $python_version is not supported"
        print_info "Supported versions: Python 3.8 - 3.12"
        exit 1
    fi

    # Check available disk space (require at least 5GB)
    local available_space=$(df . | tail -1 | awk '{print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        print_warning "Low disk space detected. At least 5GB recommended."
    fi

    # Check memory (require at least 4GB)
    if command -v free >/dev/null 2>&1; then
        local total_mem=$(free -m | awk 'NR==2{print $2}')
        if [[ $total_mem -lt 4096 ]]; then
            print_warning "Low memory detected. At least 4GB RAM recommended."
        fi
    fi

    print_success "System requirements check completed"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        if command -v apt-get >/dev/null 2>&1; then
            DISTRO="debian"
        elif command -v yum >/dev/null 2>&1; then
            DISTRO="redhat"
        elif command -v pacman >/dev/null 2>&1; then
            DISTRO="arch"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        DISTRO="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS_TYPE="windows"
        DISTRO="windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi

    print_info "Detected OS: $OS_TYPE ($DISTRO)"
}

# Install system dependencies
install_system_dependencies() {
    print_header "Installing System Dependencies"

    case $DISTRO in
        "debian")
            print_step "Installing system packages (Debian/Ubuntu)..."
            sudo apt-get update -qq

            # Essential build tools
            sudo apt-get install -y \
                build-essential \
                curl \
                wget \
                git \
                pkg-config \
                cmake \
                ninja-build \
                libssl-dev \
                libffi-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                libncurses5-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev \
                libxml2-dev \
                libxmlsec1-dev \
                liblzma-dev

            # Audio dependencies
            if [[ $INSTALL_VOICE == true ]]; then
                sudo apt-get install -y \
                    portaudio19-dev \
                    python3-pyaudio \
                    libasound2-dev \
                    libpulse-dev \
                    alsa-utils \
                    pulseaudio \
                    espeak \
                    espeak-data \
                    libespeak1 \
                    libespeak-dev \
                    festival \
                    festvox-kallpc16k \
                    ffmpeg \
                    flac \
                    sox \
                    libsox-fmt-all
            fi

            # Graphics dependencies
            sudo apt-get install -y \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                libgtk-3-dev \
                libcairo2-dev \
                libgirepository1.0-dev
            ;;

        "macos")
            print_step "Installing system packages (macOS)..."

            # Check if Homebrew is installed
            if ! command -v brew >/dev/null 2>&1; then
                print_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi

            # Update Homebrew
            brew update

            # Install essential packages
            brew install \
                cmake \
                ninja \
                pkg-config \
                openssl \
                readline \
                sqlite3 \
                xz \
                zlib \
                tcl-tk

            # Audio dependencies
            if [[ $INSTALL_VOICE == true ]]; then
                brew install \
                    portaudio \
                    espeak \
                    festival \
                    ffmpeg \
                    flac \
                    sox
            fi

            # Graphics dependencies
            brew install \
                cairo \
                pango \
                gdk-pixbuf \
                gtk+3 \
                gobject-introspection
            ;;

        *)
            print_warning "Automatic system dependency installation not supported for $DISTRO"
            print_info "Please install the following manually:"
            echo "  - Build tools (gcc, make, cmake)"
            echo "  - Audio libraries (portaudio, alsa)"
            echo "  - Graphics libraries (OpenGL, GTK)"
            echo "  - Development headers for Python"
            ;;
    esac

    print_success "System dependencies installed"
}

# Create project structure
create_project_structure() {
    print_header "Creating Project Structure"

    # Create necessary directories
    local directories=(
        "data"
        "data/models"
        "data/cache"
        "data/databases"
        "logs"
        "output"
        "output/generated"
        "output/audio"
        "output/images"
        "config"
        "temp"
        "backups"
    )

    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_info "Created directory: $dir"
        fi
    done

    # Set proper permissions
    chmod 755 data logs output config temp backups
    chmod 777 temp  # Temp directory needs write access

    print_success "Project structure created"
}

# Setup Python virtual environment
setup_virtual_environment() {
    print_header "Setting Up Python Virtual Environment"

    local venv_dir="venv"

    if [[ $FORCE_REINSTALL == true ]] && [[ -d "$venv_dir" ]]; then
        print_step "Removing existing virtual environment..."
        rm -rf "$venv_dir"
    fi

    if [[ ! -d "$venv_dir" ]]; then
        print_step "Creating virtual environment..."
        $PYTHON_CMD -m venv "$venv_dir"
    fi

    # Activate virtual environment
    source "$venv_dir/bin/activate"

    # Upgrade pip, setuptools, and wheel
    print_step "Upgrading pip and build tools..."
    python -m pip install --upgrade pip setuptools wheel

    print_success "Virtual environment ready"
}

# Install PyTorch with proper version handling
install_pytorch() {
    print_header "Installing PyTorch"

    local python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"

    # Clear pip cache to avoid issues
    python -m pip cache purge

    if [[ $INSTALL_GPU == true ]]; then
        print_step "Installing PyTorch with CUDA support..."

        # Try different PyTorch installation methods based on Python version
        if [[ "$python_version" == "3.8" ]] || [[ "$python_version" == "3.9" ]] || [[ "$python_version" == "3.10" ]] || [[ "$python_version" == "3.11" ]]; then
            # Standard installation for supported versions
            if ! python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118; then
                print_warning "CUDA installation failed, trying CPU version..."
                python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
            fi
        elif [[ "$python_version" == "3.12" ]]; then
            # Python 3.12 - try nightly builds
            print_info "Python 3.12 detected, trying nightly builds..."
            if ! python -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu118; then
                print_warning "Nightly CUDA build failed, trying CPU nightly..."
                python -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
            fi
        else
            # Python 3.13+ - try source installation or nightly
            print_warning "Python $python_version is very new, trying nightly builds..."
            if ! python -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu; then
                print_error "PyTorch installation failed for Python $python_version"
                print_info "Consider using Python 3.11 for best compatibility"
                exit 1
            fi
        fi
    else
        print_step "Installing PyTorch (CPU only)..."
        if [[ "$python_version" == "3.12" ]] || [[ "$python_version" > "3.12" ]]; then
            python -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
        else
            python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
        fi
    fi

    # Verify PyTorch installation
    if python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
        print_success "PyTorch installed successfully"
        if [[ $INSTALL_GPU == true ]]; then
            python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        fi
    else
        print_error "PyTorch installation verification failed"
        exit 1
    fi
}

# Install Python dependencies
install_python_dependencies() {
    print_header "Installing Python Dependencies"

    # Install core dependencies first
    print_step "Installing core AI/ML packages..."
    local core_packages=(
        "numpy>=1.21.0,<2.0.0"
        "scipy>=1.7.0"
        "scikit-learn>=1.0.0"
        "pandas>=1.3.0"
        "matplotlib>=3.4.0"
        "seaborn>=0.11.0"
        "pillow>=8.3.0"
        "opencv-python>=4.5.0"
        "transformers>=4.20.0"
        "tokenizers>=0.13.0"
        "datasets>=2.0.0"
        "accelerate>=0.20.0"
    )

    for package in "${core_packages[@]}"; do
        if ! python -m pip install "$package"; then
            print_warning "Failed to install $package, continuing..."
        fi
    done

    # Install web and networking packages
    print_step "Installing web and networking packages..."
    local web_packages=(
        "requests>=2.25.0"
        "aiohttp>=3.8.0"
        "beautifulsoup4>=4.9.0"
        "lxml>=4.6.0"
        "selenium>=4.0.0"
        "feedparser>=6.0.0"
        "newspaper3k>=0.2.8"
    )

    for package in "${web_packages[@]}"; do
        if ! python -m pip install "$package"; then
            print_warning "Failed to install $package, continuing..."
        fi
    done

    # Install voice-related packages if requested
    if [[ $INSTALL_VOICE == true ]]; then
        print_step "Installing voice and audio packages..."
        local voice_packages=(
            "speechrecognition>=3.8.0"
            "pyttsx3>=2.90"
            "gtts>=2.2.0"
            "librosa>=0.9.0"
            "soundfile>=0.10.0"
            "pyaudio"
            "sounddevice>=0.4.0"
            "whisper-openai"
            "vosk>=0.3.0"
        )

        for package in "${voice_packages[@]}"; do
            if ! python -m pip install "$package"; then
                print_warning "Failed to install $package (may require system dependencies)"
            fi
        done
    fi

    # Install additional packages based on requirements files
    if [[ -f "requirements.txt" ]]; then
        print_step "Installing from requirements.txt..."
        python -m pip install -r requirements.txt || print_warning "Some packages from requirements.txt failed to install"
    fi

    if [[ $INSTALL_VOICE == true ]] && [[ -f "requirements_voice.txt" ]]; then
        print_step "Installing from requirements_voice.txt..."
        python -m pip install -r requirements_voice.txt || print_warning "Some voice packages failed to install"
    fi

    # Install development dependencies if requested
    if [[ $INSTALL_DEV == true ]]; then
        print_step "Installing development packages..."
        local dev_packages=(
            "pytest>=6.0.0"
            "pytest-asyncio>=0.18.0"
            "black>=22.0.0"
            "flake8>=4.0.0"
            "mypy>=0.950"
            "jupyter>=1.0.0"
            "ipython>=7.0.0"
        )

        for package in "${dev_packages[@]}"; do
            python -m pip install "$package" || print_warning "Failed to install $package"
        done
    fi

    print_success "Python dependencies installation completed"
}

# Download and setup AI models
setup_ai_models() {
    print_header "Setting Up AI Models"

    # Create models directory
    mkdir -p data/models

    # Download spaCy models
    print_step "Downloading spaCy language models..."
    python -m spacy download en_core_web_sm || print_warning "Failed to download spaCy English model"

    # Download NLTK data
    print_step "Downloading NLTK data..."
    python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'NLTK download warning: {e}')
" || print_warning "NLTK data download had issues"

    print_success "AI models setup completed"
}

# Create configuration files
create_configuration_files() {
    print_header "Creating Configuration Files"

    # Main configuration file
    if [[ ! -f "config.yaml" ]] || [[ $FORCE_REINSTALL == true ]]; then
        print_step "Creating main configuration file..."
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
EOF
        print_success "Created config.yaml"
    fi

    # Voice configuration file
    if [[ ! -f "config/voice_config.yaml" ]] || [[ $FORCE_REINSTALL == true ]]; then
        print_step "Creating voice configuration file..."
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

commands:
  enabled: true
  custom_commands_file: "config/custom_commands.yaml"
EOF
        print_success "Created config/voice_config.yaml"
    fi

    print_success "Configuration files created"
}

# Initialize database
initialize_database() {
    print_header "Initializing Database"

    print_step "Creating database schema..."
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

# Insert default preferences
cursor.execute('''
INSERT OR IGNORE INTO user_preferences (key, value) VALUES
('voice_enabled', 'true'),
('debug_mode', 'false'),
('auto_save', 'true')
''')

conn.commit()
conn.close()
print('Database initialized successfully')
" || print_warning "Database initialization had issues"

    print_success "Database initialized"
}

# Create startup scripts
create_startup_scripts() {
    print_header "Creating Startup Scripts"

    # Main startup script
    cat > start_shock2.sh << 'EOF'
#!/bin/bash
# Shock2 AI System Startup Script

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the main application
echo "Starting Shock2 AI System..."
python main.py "$@"
EOF
    chmod +x start_shock2.sh

    # Voice interface startup script
    cat > start_voice_interface.sh << 'EOF'
#!/bin/bash
# Shock2 Voice Interface Startup Script

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the voice interface
echo "Starting Shock2 Voice Interface..."
python shock2_voice_interface.py "$@"
EOF
    chmod +x start_voice_interface.sh

    # Development startup script
    if [[ $INSTALL_DEV == true ]]; then
        cat > start_dev.sh << 'EOF'
#!/bin/bash
# Shock2 Development Environment Startup Script

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DEBUG=true

# Start Jupyter notebook
echo "Starting development environment..."
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
EOF
        chmod +x start_dev.sh
    fi

    print_success "Startup scripts created"
}

# Run system tests
run_system_tests() {
    print_header "Running System Tests"

    print_step "Testing Python imports..."
    python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import torch
    print(f'? PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'? CUDA available: {torch.cuda.device_count()} devices')
    else:
        print('??  CUDA not available (CPU mode)')
except ImportError as e:
    print(f'? PyTorch import failed: {e}')

try:
    import transformers
    print(f'? Transformers {transformers.__version__}')
except ImportError as e:
    print(f'? Transformers import failed: {e}')

try:
    import cv2
    print(f'? OpenCV {cv2.__version__}')
except ImportError as e:
    print(f'? OpenCV import failed: {e}')

try:
    import numpy as np
    print(f'? NumPy {np.__version__}')
except ImportError as e:
    print(f'? NumPy import failed: {e}')
"

    if [[ $INSTALL_VOICE == true ]]; then
        print_step "Testing voice components..."
        python -c "
try:
    import speech_recognition as sr
    print('? SpeechRecognition available')
except ImportError:
    print('? SpeechRecognition not available')

try:
    import pyttsx3
    engine = pyttsx3.init()
    print('? Text-to-Speech engine initialized')
except ImportError:
    print('? Text-to-Speech not available')
except Exception as e:
    print(f'??  TTS initialization warning: {e}')
"
    fi

    print_step "Testing database connection..."
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

# Main installation function
main() {
    print_header "Shock2 AI System - Complete Setup"
    print_info "This script will install and configure the complete Shock2 AI system"

    # Parse command line arguments
    parse_args "$@"

    # Show configuration
    echo ""
    print_info "Installation Configuration:"
    echo "  Python Command: $PYTHON_CMD"
    echo "  GPU Support: $INSTALL_GPU"
    echo "  Voice Interface: $INSTALL_VOICE"
    echo "  Development Mode: $INSTALL_DEV"
    echo "  Minimal Install: $MINIMAL_INSTALL"
    echo "  Force Reinstall: $FORCE_REINSTALL"
    echo ""

    # Confirm installation
    if [[ $FORCE_REINSTALL == false ]]; then
        read -p "Continue with installation? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installation cancelled"
            exit 0
        fi
    fi

    # Run installation steps
    check_system_requirements
    detect_os
    install_system_dependencies
    create_project_structure
    setup_virtual_environment
    install_pytorch
    install_python_dependencies

    if [[ $MINIMAL_INSTALL == false ]]; then
        setup_ai_models
        initialize_database
    fi

    create_configuration_files
    create_startup_scripts
    run_system_tests

    # Final success message
    print_header "Installation Complete!"
    print_success "Shock2 AI System has been successfully installed"

    echo ""
    print_info "Quick Start:"
    echo "  1. Start main system:     ./start_shock2.sh"
    echo "  2. Start voice interface: ./start_voice_interface.sh"
    if [[ $INSTALL_DEV == true ]]; then
        echo "  3. Start development:     ./start_dev.sh"
    fi
    echo ""

    print_info "Configuration files:"
    echo "  - Main config: config.yaml"
    echo "  - Voice config: config/voice_config.yaml"
    echo "  - Database: data/databases/shock2.db"
    echo ""

    print_info "For help and documentation, check the README files"
    print_success "Setup completed successfully! ?"
}

# Run main function with all arguments
main "$@"
