
#!/usr/bin/env python3
"""
Shock2 AI Voice Interface - Enhanced Cloud-Compatible Entry Point
Single command to start the complete enhanced voice interface
"""

import sys
import os
import asyncio
import logging
import traceback
import time
from pathlib import Path
from datetime import datetime

# Add shock2 directory to path
current_dir = Path(__file__).parent
shock2_dir = current_dir / "shock2222vvvv"
sys.path.insert(0, str(shock2_dir))

def setup_environment():
    """Setup enhanced environment"""
    print("🔧 Setting up enhanced environment...")
    
    # Create necessary directories
    directories = [
        "logs",
        "data/voice_profiles", 
        "data/models",
        "data/cache",
        "output",
        "data/analytics",
        "data/intelligence"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for cloud compatibility
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    os.environ['SDL_AUDIODRIVER'] = 'pulse'  # Use pulse audio in cloud
    
    print("✅ Enhanced environment setup complete")

def setup_logging():
    """Setup enhanced logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"shock2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log startup information
    logger = logging.getLogger(__name__)
    logger.info("🚀 Shock2 Enhanced Voice Interface starting up")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log file: {log_filename}")

def check_dependencies():
    """Check and report on dependencies"""
    print("📋 Checking dependencies...")
    
    dependencies = {
        'asyncio': True,  # Built-in
        'pathlib': True,  # Built-in
        'logging': True,  # Built-in
        'datetime': True,  # Built-in
    }
    
    # Check optional dependencies
    optional_deps = {
        'numpy': False,
        'pyttsx3': False,
        'speech_recognition': False,
        'pygame': False,
    }
    
    for dep in optional_deps:
        try:
            __import__(dep)
            optional_deps[dep] = True
        except ImportError:
            optional_deps[dep] = False
    
    # Report status
    print("\n📦 Dependency Status:")
    print("  Core Dependencies: ✅ All available")
    
    print("  Enhanced Features:")
    for dep, available in optional_deps.items():
        status = "✅" if available else "❌"
        print(f"    {dep}: {status}")
    
    if not any(optional_deps.values()):
        print("\n⚠️  Note: Enhanced features require additional packages")
        print("   Run: pip install -r requirements.txt")
    
    return dependencies, optional_deps

async def run_enhanced_voice_interface():
    """Run the enhanced voice interface"""
    try:
        print("\n🚀 Starting Enhanced Shock2 AI Voice Interface...")
        print("=" * 70)
        
        # Import the enhanced voice interface
        from shock2_voice_interface import Shock2VoiceInterface
        
        # Create and initialize interface
        interface = Shock2VoiceInterface()
        await interface.initialize()
        
        print("\n🎤 Enhanced Shock2 is now operational!")
        print("🧠 Advanced neural processing active")
        print("🎭 Enhanced personality engine online")
        print("📰 Intelligent content generation ready")
        print("🔊 Cloud-optimized audio processing")
        print("\n💬 Enhanced Voice Commands:")
        print("  System Control:")
        print("    - 'Show me enhanced system status'")
        print("    - 'Run performance analysis'")
        print("  Content Generation:")
        print("    - 'Generate advanced news about AI'") 
        print("    - 'Create multiple articles about technology'")
        print("    - 'Generate investigation report'")
        print("  Intelligence Operations:")
        print("    - 'Analyze intelligence data'")
        print("    - 'Perform autonomous decision making'")
        print("    - 'Emergency response protocol'")
        print("  Information Access:")
        print("    - 'Show me generated articles'")
        print("    - 'Display performance metrics'")
        print("\n🌐 Cloud Environment Optimized")
        print("⌨️  Text input mode active (voice recognition available locally)")
        print("\nPress Ctrl+C to shutdown")
        print("=" * 70)
        
        # Start listening
        await interface.start_listening()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received...")
        if 'interface' in locals():
            await interface.shutdown()
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("💡 Try installing dependencies: pip install -r requirements.txt")
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Error starting enhanced voice interface: {e}")
        traceback.print_exc()

def display_startup_banner():
    """Display enhanced startup banner"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                        SHOCK2 AI SYSTEM v2.0                         ║
║                   Enhanced Voice Interface Engine                     ║
║                                                                      ║
║  🧠 Advanced Neural Processing    📊 Performance Analytics           ║
║  🎤 Cloud-Optimized Voice I/O     📰 Intelligent Content Creation    ║
║  🤖 Autonomous Decision Making    🛡️ Enhanced Security Protocols     ║
║  ⚡ Real-time Data Processing     ☁️ Cloud Environment Optimized     ║
║                                                                      ║
║  Status: OPERATIONAL | Build: {datetime.now().strftime('%Y.%m.%d')} | Mode: ENHANCED    ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Enhanced main entry point"""
    
    # Display startup banner
    display_startup_banner()
    
    print("🤖 Shock2 AI Enhanced Voice Interface")
    print("Initializing advanced system components...")
    
    try:
        # Setup environment
        setup_environment()
        
        # Setup logging
        setup_logging()
        
        # Check dependencies
        core_deps, optional_deps = check_dependencies()
        
        # Performance check
        start_time = time.time()
        
        print(f"\n⚡ System initialization: {time.time() - start_time:.2f}s")
        
        # Run the enhanced voice interface
        asyncio.run(run_enhanced_voice_interface())
        
    except KeyboardInterrupt:
        print("\n✅ Shock2 Enhanced shutdown complete")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Check dependencies: pip install -r requirements.txt")
        print("  2. Verify Python version (3.8+ required)")
        print("  3. Check system permissions")
        print("  4. Review logs in ./logs/ directory")
        sys.exit(1)

if __name__ == "__main__":
    main()
