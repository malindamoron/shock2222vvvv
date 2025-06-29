
#!/usr/bin/env python3
"""
Shock2 AI Voice Interface - Simplified Main Entry Point
Single command to start the complete voice interface with visual face
"""

import sys
import os
import asyncio
import logging
import traceback
from pathlib import Path

# Add shock2 directory to path
current_dir = Path(__file__).parent
shock2_dir = current_dir / "shock2222vvvv"
sys.path.insert(0, str(shock2_dir))

def setup_environment():
    """Setup basic environment"""
    # Create necessary directories
    directories = [
        "logs",
        "data/voice_profiles", 
        "data/models",
        "data/cache",
        "output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup complete")

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/shock2.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

async def run_voice_interface():
    """Run the voice interface"""
    try:
        print("🚀 Starting Shock2 AI Voice Interface...")
        print("=" * 50)
        
        # Import the voice interface
        from shock2_voice_interface import Shock2VoiceInterface
        
        # Create and initialize interface
        interface = Shock2VoiceInterface()
        await interface.initialize()
        
        print("\n🎤 Shock2 is now listening for voice commands...")
        print("🎭 Visual face animation active")
        print("🔊 Speak naturally to interact with Shock2")
        print("\nExample commands:")
        print("- 'Show me system status'")
        print("- 'Generate news about AI'") 
        print("- 'Activate stealth mode'")
        print("- 'What's your current state?'")
        print("\nPress Ctrl+C to shutdown")
        print("=" * 50)
        
        # Start listening
        await interface.start_listening()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Shock2...")
        if 'interface' in locals():
            await interface.shutdown()
    except Exception as e:
        print(f"❌ Error starting voice interface: {e}")
        traceback.print_exc()

def main():
    """Main entry point"""
    print("🤖 Shock2 AI System - Voice Interface")
    print("Initializing system components...")
    
    setup_environment()
    setup_logging()
    
    # Run the voice interface
    try:
        asyncio.run(run_voice_interface())
    except KeyboardInterrupt:
        print("\n✅ Shock2 shutdown complete")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
