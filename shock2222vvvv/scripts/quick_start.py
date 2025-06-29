#!/usr/bin/env python3
"""
Quick Start Script for Shock2 Voice Interface
Simplified startup with automatic configuration
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

def setup_environment():
    """Setup the environment and paths"""
    # Get current directory
    current_dir = os.getcwd()
    
    # Add current directory to Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Create basic directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/voice_profiles', exist_ok=True)
    os.makedirs('data/security/voice_prints', exist_ok=True)
    os.makedirs('data/personalities', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)
    
    print(f"✅ Environment setup complete")
    print(f"📁 Working directory: {current_dir}")

def create_default_config():
    """Create a default configuration"""
    return {
        'logging': {
            'level': 'INFO',
            'file': 'logs/voice_interface.log'
        },
        'voice_recognition': {
            'engine': 'whisper',
            'model': 'base',
            'language': 'en',
            'energy_threshold': 300,
            'pause_threshold': 0.8
        },
        'text_to_speech': {
            'engine': 'pyttsx3',
            'voice_id': None,
            'rate': 200,
            'volume': 0.9
        },
        'wake_word': {
            'enabled': True,
            'keywords': ['shock', 'shock2'],
            'sensitivity': 0.5
        },
        'voice_cloning': {
            'enabled': True,
            'profiles_dir': 'data/voice_profiles',
            'model_path': 'data/models/voice_cloning'
        },
        'authentication': {
            'enabled': True,
            'voice_prints_dir': 'data/security/voice_prints',
            'threshold': 0.8
        },
        'persona_management': {
            'enabled': True,
            'personalities_dir': 'data/personalities',
            'default_personality': 'shock2_default'
        },
        'animation': {
            'lip_sync': {
                'enabled': True,
                'model_path': 'data/models/lipsync'
            },
            'face_engine': {
                'enabled': True,
                'model_path': 'data/models/face'
            }
        },
        'integration': {
            'max_workers': 4,
            'audio_buffer_size': 1024,
            'processing_timeout': 30.0,
            'health_check_interval': 5.0
        }
    }

async def simple_voice_test():
    """Simple voice interface test"""
    print("\n🎤 Starting Simple Voice Test...")
    
    try:
        # Try to import and test basic components
        print("📦 Testing imports...")
        
        # Test basic speech recognition
        try:
            import speech_recognition as sr
            print("✅ Speech Recognition: Available")
        except ImportError:
            print("❌ Speech Recognition: Not available (install speech_recognition)")
        
        # Test text-to-speech
        try:
            import pyttsx3
            print("✅ Text-to-Speech: Available")
        except ImportError:
            print("❌ Text-to-Speech: Not available (install pyttsx3)")
        
        # Test audio
        try:
            import pyaudio
            print("✅ Audio System: Available")
        except ImportError:
            print("❌ Audio System: Not available (install pyaudio)")
        
        print("\n🔧 Basic voice test complete!")
        print("💡 Install missing dependencies with: pip install -r requirements_voice.txt")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

async def main():
    """Main entry point for quick start"""
    print("🚀 Shock2 Voice Interface - Quick Start")
    print("="*50)
    
    # Setup environment
    setup_environment()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/quick_start.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Run simple test
        await simple_voice_test()
        
        print("\n" + "="*50)
        print("🎯 Next Steps:")
        print("1. Install dependencies: pip install -r requirements_voice.txt")
        print("2. Run full system: python scripts/start_voice_interface.py")
        print("3. Or continue with component testing")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Quick start failed: {e}")
        print(f"\n❌ Quick start failed: {e}")
        print("💡 Try installing dependencies first")

if __name__ == "__main__":
    asyncio.run(main())
