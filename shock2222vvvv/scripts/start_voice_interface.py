#!/usr/bin/env python3
"""
Shock2 Voice Interface Startup Script
Complete system initialization and startup
"""

import asyncio
import logging
import os
import sys
import yaml
import argparse
from pathlib import Path

# More robust path setup
def get_project_root():
    """Get the project root directory"""
    try:
        # Try using __file__ if available
        if '__file__' in globals():
            return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except:
        pass
    
    # Fallback to current working directory
    current_dir = os.getcwd()
    
    # Look for shock2 directory or main.py to identify project root
    if os.path.exists(os.path.join(current_dir, 'shock2')):
        return current_dir
    elif os.path.exists(os.path.join(current_dir, '..', 'shock2')):
        return os.path.dirname(current_dir)
    else:
        # Default to current directory
        return current_dir

# Add shock2 to path
project_root = get_project_root()
sys.path.insert(0, project_root)

print(f"🔧 Project root: {project_root}")
print(f"🔧 Python path: {sys.path[0]}")

from shock2.voice.integration.voice_orchestrator import Shock2VoiceOrchestrator
from shock2.voice.integration.system_coordinator import Shock2SystemCoordinator

def setup_logging(config: dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', 'logs/voice_interface.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_directories(config: dict):
    """Create necessary directories"""
    directories = [
        config.get('voice_cloning', {}).get('profiles_dir', 'data/voice_profiles'),
        config.get('authentication', {}).get('voice_prints_dir', 'data/security/voice_prints'),
        config.get('persona_management', {}).get('personalities_dir', 'data/personalities'),
        'logs',
        'data/models',
        'data/cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Shock2 Voice Interface')
    parser.add_argument('--config', '-c', default='config/voice_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio streaming (for testing)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override debug mode
        if args.debug:
            config['logging'] = config.get('logging', {})
            config['logging']['level'] = 'DEBUG'
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        # Create directories
        create_directories(config)
        
        logger.info("🚀 Starting Shock2 Voice Interface...")
        
        # Initialize voice orchestrator
        voice_orchestrator = Shock2VoiceOrchestrator(config)
        
        # Initialize system coordinator
        coordinator_config = config.get('integration', {})
        system_coordinator = Shock2SystemCoordinator(coordinator_config)
        
        # Initialize voice orchestrator
        await voice_orchestrator.initialize()
        
        # Initialize system coordinator with voice orchestrator
        await system_coordinator.initialize(
            voice_orchestrator,
            voice_orchestrator.system_manager,
            voice_orchestrator.core_orchestrator,
            voice_orchestrator.autonomous_controller
        )
        
        # Print startup information
        print("\n" + "="*60)
        print("🤖 SHOCK2 VOICE INTERFACE OPERATIONAL")
        print("="*60)
        print("🎤 Voice Recognition: ACTIVE")
        print("🧠 AI Personalities: LOADED")
        print("🔐 Security Systems: ENABLED")
        print("🎭 Face Animation: READY")
        print("🗣️ Voice Synthesis: OPERATIONAL")
        print("⚡ Real-time Processing: ACTIVE")
        print("="*60)
        print("\n💬 Say 'Shock' or 'Shock2' to activate voice commands")
        print("🎯 Available personalities:")
        
        if voice_orchestrator.persona_manager:
            personalities = voice_orchestrator.persona_manager.get_personality_profiles()
            for personality in personalities:
                print(f"   • {personality.name} - {personality.description}")
        
        print("\n📊 System Status:")
        status = voice_orchestrator.get_system_status()
        print(f"   • System State: {status['system_state']}")
        print(f"   • Active Components: {sum(status['component_status'].values())}")
        print(f"   • Audio Latency: {status['performance_metrics']['audio_latency']:.3f}s")
        
        print("\n🔧 Controls:")
        print("   • Ctrl+C: Graceful shutdown")
        print("   • Voice commands: Natural language interaction")
        print("   • Web interface: http://localhost:8080 (if enabled)")
        print("\n" + "="*60 + "\n")
        
        # Wait for shutdown
        try:
            await voice_orchestrator.wait_for_shutdown()
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown signal received")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            if 'voice_orchestrator' in locals():
                await voice_orchestrator.shutdown()
            if 'system_coordinator' in locals():
                await system_coordinator.shutdown()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        print("\n✅ Shock2 Voice Interface shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
