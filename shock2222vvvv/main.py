#!/usr/bin/env python3
"""
Shock2 AI News System - Main Entry Point
Advanced autonomous news generation and publishing system
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add the shock2 package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from shock2.core.system_manager import Shock2SystemManager
from shock2.config.settings import load_config
from shock2.config.logging import setup_logging
from shock2.utils.exceptions import Shock2Exception

def main():
    """Main entry point for Shock2 AI System"""
    try:
        print("🚀 Initializing Shock2 AI News System...")
        print("=" * 60)
        
        # Load configuration
        config = load_config()
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("🔧 Shock2 AI System starting up...")
        
        # Initialize and run system manager
        system_manager = Shock2SystemManager(config)
        
        # Run the system
        asyncio.run(system_manager.run())
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received")
        logger.info("System shutdown requested by user")
    except Shock2Exception as e:
        print(f"❌ Shock2 Error: {e}")
        logger.error(f"Shock2 specific error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        print("🔚 Shock2 AI System shutdown complete")
        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()
