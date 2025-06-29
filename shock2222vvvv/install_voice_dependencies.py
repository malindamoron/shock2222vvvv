#!/usr/bin/env python3
"""
Shock2 Voice Interface Dependencies Installer
Installs all required packages for voice interface functionality
"""

import subprocess
import sys
import os
import platform

def run_command(command):
    """Run shell command and return result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    
    print("🔧 Installing system dependencies...")
    
    if system == "linux":
        # Ubuntu/Debian
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y portaudio19-dev python3-pyaudio",
            "sudo apt-get install -y espeak espeak-data libespeak1 libespeak-dev",
            "sudo apt-get install -y flac",
            "sudo apt-get install -y ffmpeg",
            "sudo apt-get install -y libsndfile1-dev"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            success, stdout, stderr = run_command(cmd)
            if not success:
                print(f"Warning: Command failed: {cmd}")
                print(f"Error: {stderr}")
    
    elif system == "darwin":  # macOS
        commands = [
            "brew install portaudio",
            "brew install espeak",
            "brew install flac",
            "brew install ffmpeg"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            success, stdout, stderr = run_command(cmd)
            if not success:
                print(f"Warning: Command failed: {cmd}")
    
    elif system == "windows":
        print("Windows detected. Please install:")
        print("1. Microsoft Visual C++ Build Tools")
        print("2. PyAudio wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
        print("3. eSpeak from: http://espeak.sourceforge.net/download.html")

def install_python_packages():
    """Install Python packages"""
    print("📦 Installing Python packages...")
    
    # Install basic requirements first
    basic_packages = [
        "wheel",
        "setuptools",
        "pip --upgrade"
    ]
    
    for package in basic_packages:
        cmd = f"{sys.executable} -m pip install {package}"
        print(f"Installing: {package}")
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"Failed to install {package}: {stderr}")
    
    # Install from requirements file
    requirements_file = "requirements_voice.txt"
    if os.path.exists(requirements_file):
        cmd = f"{sys.executable} -m pip install -r {requirements_file}"
        print(f"Installing from {requirements_file}...")
        success, stdout, stderr = run_command(cmd)
        if not success:
            print(f"Failed to install from requirements: {stderr}")
            return False
    else:
        print(f"Requirements file {requirements_file} not found!")
        return False
    
    return True

def download_models():
    """Download required models"""
    print("🧠 Downloading AI models...")
    
    # Download spaCy models
    spacy_models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    
    for model in spacy_models:
        cmd = f"{sys.executable} -m spacy download {model}"
        print(f"Downloading spaCy model: {model}")
        success, stdout, stderr = run_command(cmd)
        if success:
            print(f"✅ Downloaded {model}")
            break  # Only need one model to work
        else:
            print(f"❌ Failed to download {model}")
    
    print("✅ Model download complete")

def verify_installation():
    """Verify that all components are working"""
    print("🔍 Verifying installation...")
    
    test_imports = [
        "speech_recognition",
        "pyttsx3",
        "sounddevice",
        "webrtcvad",
        "librosa",
        "spacy",
        "transformers",
        "torch",
        "cv2",
        "pygame",
        "numpy",
        "scipy"
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All components verified successfully!")
        return True

def main():
    """Main installation process"""
    print("🚀 Shock2 Voice Interface Installation")
    print("=" * 50)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Install Python packages
    if not install_python_packages():
        print("❌ Failed to install Python packages")
        sys.exit(1)
    
    # Download models
    download_models()
    
    # Verify installation
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        print("\nTo start the Shock2 Voice Interface, run:")
        print("python shock2_voice_interface.py")
    else:
        print("\n❌ Installation verification failed")
        print("Please check the error messages above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
