"""
Shock2 Configuration Management
Centralized configuration system with YAML support and environment variable overrides
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "sqlite"
    database: str = "data/databases/shock2.db"
    pool_size: int = 10
    timeout: int = 30


@dataclass
class NeuralConfig:
    """Neural network configuration"""
    model_name: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    max_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.9
    batch_size: int = 4
    cache_dir: str = "data/models"


@dataclass
class IntelligenceConfig:
    """Intelligence gathering configuration"""
    max_concurrent_requests: int = 10
    request_delay: float = 1.0
    timeout: int = 30
    retry_attempts: int = 3
    user_agents: list = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ])


@dataclass
class StealthConfig:
    """Stealth and evasion configuration"""
    enabled: bool = True
    detection_evasion: bool = True
    signature_masking: bool = True
    pattern_randomization: bool = True
    proxy_rotation: bool = False
    anonymization_level: str = "high"


@dataclass
class GenerationConfig:
    """Content generation configuration"""
    article_types: list = field(default_factory=lambda: ["breaking_news", "analysis", "summary", "opinion"])
    max_articles_per_cycle: int = 20
    quality_threshold: float = 0.7
    seo_optimization: bool = True
    readability_optimization: bool = True


@dataclass
class PublishingConfig:
    """Publishing configuration"""
    output_directory: str = "output"
    formats: list = field(default_factory=lambda: ["markdown", "json", "html"])
    organize_by_category: bool = True
    include_metadata: bool = True
    backup_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_port: int = 8080
    log_level: str = "INFO"
    log_directory: str = "logs"
    performance_tracking: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.05,
        "memory_usage": 0.8,
        "cpu_usage": 0.8
    })


@dataclass
class Shock2Config:
    """Main Shock2 configuration"""
    environment: str = "production"
    debug: bool = False
    cycle_interval: int = 300

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    stealth: StealthConfig = field(default_factory=StealthConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    publishing: PublishingConfig = field(default_factory=PublishingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    data_sources: list = field(default_factory=lambda: [
        "https://rss.cnn.com/rss/edition.rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.reuters.com/reuters/topNews",
        "https://feeds.npr.org/1001/rss.xml",
        "https://www.theguardian.com/world/rss"
    ])

    api_keys: Dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager with environment variable support"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self.config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from file and environment variables"""
        # Load from YAML file
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = {}

        # Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)

        # Create config object
        self.config = self._create_config_object(config_data)

    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_mappings = {
            'SHOCK2_DEBUG': ('debug', bool),
            'SHOCK2_LOG_LEVEL': ('monitoring.log_level', str),
            'SHOCK2_DATABASE_PATH': ('database.database', str),
            'SHOCK2_MODEL_CACHE': ('neural.cache_dir', str),
            'SHOCK2_OUTPUT_DIR': ('publishing.output_directory', str),
            'SHOCK2_CYCLE_INTERVAL': ('cycle_interval', int),
            'SHOCK2_MAX_ARTICLES': ('generation.max_articles_per_cycle', int),
        }

        for env_var, (config_path, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert value to appropriate type
                if value_type == bool:
                    env_value = env_value.lower() in ('true', '1', 'yes', 'on')
                elif value_type == int:
                    env_value = int(env_value)
                elif value_type == float:
                    env_value = float(env_value)

                # Set nested config value
                self._set_nested_value(config_data, config_path, env_value)

        return config_data

    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _create_config_object(self, config_data: Dict[str, Any]) -> Shock2Config:
        """Create configuration object from data"""
        # Create sub-configs
        database_config = DatabaseConfig(**config_data.get('database', {}))
        neural_config = NeuralConfig(**config_data.get('neural', {}))
        intelligence_config = IntelligenceConfig(**config_data.get('intelligence', {}))
        stealth_config = StealthConfig(**config_data.get('stealth', {}))
        generation_config = GenerationConfig(**config_data.get('generation', {}))
        publishing_config = PublishingConfig(**config_data.get('publishing', {}))
        monitoring_config = MonitoringConfig(**config_data.get('monitoring', {}))

        # Create main config
        main_config_data = {k: v for k, v in config_data.items()
                            if k not in ['database', 'neural', 'intelligence', 'stealth',
                                         'generation', 'publishing', 'monitoring']}

        return Shock2Config(
            database=database_config,
            neural=neural_config,
            intelligence=intelligence_config,
            stealth=stealth_config,
            generation=generation_config,
            publishing=publishing_config,
            monitoring=monitoring_config,
            **main_config_data
        )

    def get_config(self) -> Shock2Config:
        """Get configuration object"""
        return self.config

    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()

    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = config_path or self.config_path

        # Convert config object to dictionary
        config_dict = self._config_to_dict(self.config)

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save to YAML file
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _config_to_dict(self, config: Shock2Config) -> Dict[str, Any]:
        """Convert config object to dictionary"""
        return {
            'environment': config.environment,
            'debug': config.debug,
            'cycle_interval': config.cycle_interval,
            'database': {
                'type': config.database.type,
                'database': config.database.database,
                'pool_size': config.database.pool_size,
                'timeout': config.database.timeout
            },
            'neural': {
                'model_name': config.neural.model_name,
                'device': config.neural.device,
                'max_length': config.neural.max_length,
                'temperature': config.neural.temperature,
                'top_p': config.neural.top_p,
                'batch_size': config.neural.batch_size,
                'cache_dir': config.neural.cache_dir
            },
            'intelligence': {
                'max_concurrent_requests': config.intelligence.max_concurrent_requests,
                'request_delay': config.intelligence.request_delay,
                'timeout': config.intelligence.timeout,
                'retry_attempts': config.intelligence.retry_attempts,
                'user_agents': config.intelligence.user_agents
            },
            'stealth': {
                'enabled': config.stealth.enabled,
                'detection_evasion': config.stealth.detection_evasion,
                'signature_masking': config.stealth.signature_masking,
                'pattern_randomization': config.stealth.pattern_randomization,
                'proxy_rotation': config.stealth.proxy_rotation,
                'anonymization_level': config.stealth.anonymization_level
            },
            'generation': {
                'article_types': config.generation.article_types,
                'max_articles_per_cycle': config.generation.max_articles_per_cycle,
                'quality_threshold': config.generation.quality_threshold,
                'seo_optimization': config.generation.seo_optimization,
                'readability_optimization': config.generation.readability_optimization
            },
            'publishing': {
                'output_directory': config.publishing.output_directory,
                'formats': config.publishing.formats,
                'organize_by_category': config.publishing.organize_by_category,
                'include_metadata': config.publishing.include_metadata,
                'backup_enabled': config.publishing.backup_enabled
            },
            'monitoring': {
                'enabled': config.monitoring.enabled,
                'metrics_port': config.monitoring.metrics_port,
                'log_level': config.monitoring.log_level,
                'log_directory': config.monitoring.log_directory,
                'performance_tracking': config.monitoring.performance_tracking,
                'alert_thresholds': config.monitoring.alert_thresholds
            },
            'data_sources': config.data_sources,
            'api_keys': config.api_keys
        }


# Global config instance
_config_manager = None


def load_config(config_path: Optional[str] = None) -> Shock2Config:
    """Load configuration (singleton pattern)"""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_path)

    return _config_manager.get_config()


def get_config() -> Shock2Config:
    """Get current configuration"""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager.get_config()


def reload_config():
    """Reload configuration from file"""
    global _config_manager

    if _config_manager is not None:
        _config_manager.reload_config()


def save_config(config_path: Optional[str] = None):
    """Save current configuration"""
    global _config_manager

    if _config_manager is not None:
        _config_manager.save_config(config_path)
