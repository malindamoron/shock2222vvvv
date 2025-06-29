"""
Shock2 AI News System
Advanced autonomous news generation and publishing platforms
"""

__version__ = "2.0.0"
__author__ = "Shock2 AI Team"
__description__ = "Autonomous AI-powered news generation system"

# Core imports for easy access
from .core.system_manager import Shock2SystemManager
from .config.settings import load_config
from .utils.exceptions import Shock2Exception

# Package metadata
__all__ = [
    "Shock2SystemManager",
    "load_config", 
    "Shock2Exception",
    "__version__",
    "__author__",
    "__description__"
]

# System constants
SYSTEM_NAME = "Shock2 AI"
SYSTEM_VERSION = __version__
SYSTEM_CODENAME = "Autonomous News Intelligence"

# Feature flags
FEATURES = {
    "neural_processing": True,
    "stealth_mode": True,
    "autonomous_operation": True,
    "real_time_generation": True,
    "multi_source_intelligence": True,
    "competitive_analysis": True,
    "viral_optimization": True,
    "detection_evasion": True
}

def get_system_info():
    """Get system information"""
    return {
        "name": SYSTEM_NAME,
        "version": SYSTEM_VERSION,
        "codename": SYSTEM_CODENAME,
        "features": FEATURES,
        "author": __author__,
        "description": __description__
    }
