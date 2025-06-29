"""
Shock2 Logging Configuration
Advanced logging system with file rotation and structured output
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class Shock2Logger:
    """Advanced logging system for Shock2"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loggers = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config.get('log_directory', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'shock2.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Structured JSON log handler
        json_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, 'shock2_structured.log'),
            maxBytes=10*1024*1024,
            backupCount=3
        )
        json_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(json_handler)
        
        # Component-specific loggers
        self._setup_component_loggers(log_dir)
    
    def _setup_component_loggers(self, log_dir: str):
        """Setup component-specific loggers"""
        components = [
            'shock2.core',
            'shock2.neural',
            'shock2.intelligence',
            'shock2.generation',
            'shock2.stealth',
            'shock2.voice',
            'shock2.publishing',
            'shock2.monitoring'
        ]
        
        for component in components:
            logger = logging.getLogger(component)
            
            # Component-specific file handler
            component_name = component.split('.')[-1]
            handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, f'{component_name}.log'),
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            self.loggers[component] = logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger for specific component"""
        return logging.getLogger(name)


def setup_logging(config: Optional[Dict[str, Any]] = None) -> Shock2Logger:
    """Setup logging system"""
    if config is None:
        config = {
            'log_level': 'INFO',
            'log_directory': 'logs'
        }
    
    return Shock2Logger(config)
