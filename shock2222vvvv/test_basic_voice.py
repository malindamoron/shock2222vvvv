#!/usr/bin/env python3
"""
Basic Voice Interface Test
Test individual components without full system
"""

import os
import sys
import asyncio
import logging

def test_imports():
    """Test if required modules can be imported"""
    results = {}
    
    # Test core Python modules
    try:
        import asyncio
        results['asyncio'] = True
    except ImportError:
        results['asyncio'] = False
    
    # Test voice recognition
    try:
        import speech_recognition as sr
        results['speech_recognition'] = True
    except ImportError:
        results['speech_recognition'] = False
    
    # Test text-to-speech
    try:
        import pyttsx3
        results['pyttsx3'] = True
    except ImportError:
        results['pyttsx3'] = False
    
    # Test audio
    try:
        import pyaudio
        results['pyaudio'] = True
    except ImportError:
        results['pyaudio'] = False
    
    # Test ML libraries
    try:
        import numpy as np
        results['numpy'] = True
    except ImportError:
        results['numpy'] = False
    
    try:
        import torch
        results['torch'] = True
    except ImportError:
        results['torch'] = False
    
    return results

def test_basic_tts():
    """Test basic text-to-speech"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say("Shock2 voice interface test successful")
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"TTS test failed: {e}")
        return False

def test_microphone():
    """Test microphone access"""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Testing microphone access...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("✅ Microphone access successful")
            return True
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🧪 Shock2 Voice Interface - Basic Tests")
    print("="*50)
    
    # Test imports
    print("\n📦 Testing Module Imports:")
    results = test_imports()
    
    for module, available in results.items():
        status = "✅" if available else "❌"
        print(f"   {status} {module}")
    
    # Test microphone
    print("\n🎤 Testing Audio Input:")
    if results.get('speech_recognition') and results.get('pyaudio'):
        test_microphone()
    else:
        print("❌ Cannot test microphone - missing dependencies")
    
    # Test TTS
    print("\n🗣️ Testing Text-to-Speech:")
    if results.get('pyttsx3'):
        if test_basic_tts():
            print("✅ TTS test successful")
        else:
            print("❌ TTS test failed")
    else:
        print("❌ Cannot test TTS - missing pyttsx3")
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Summary:")
    total_tests = len(results)
    passed_tests = sum(results.values())
    print(f"   Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Ready for full voice interface.")
    else:
        print("⚠️ Some tests failed. Install missing dependencies:")
        print("   pip install -r requirements_voice.txt")
    
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
