#!/usr/bin/env python3
"""
Logging Configuration System
Environment-specific logging configuration for the video synthesis pipeline.
Supports development, production, and testing environments with different log levels,
destinations, and retention policies.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class LogDestination(Enum):
    """Log output destinations."""
    CONSOLE = "console"
    FILE = "file"
    REMOTE = "remote"
    SYSLOG = "syslog"


@dataclass
class LogRetentionPolicy:
    """Log retention configuration."""
    max_file_size_mb: int = 100
    max_files_per_component: int = 10
    days_to_keep: int = 30
    compress_old_logs: bool = True


@dataclass
class LogFormatConfig:
    """Log format configuration."""
    include_timestamp: bool = True
    include_session_id: bool = True
    include_correlation_id: bool = True
    include_memory_usage: bool = True
    include_execution_time: bool = True
    json_format: bool = True
    pretty_print: bool = False


@dataclass
class ComponentLogConfig:
    """Per-component logging configuration."""
    log_level: str = "INFO"
    enabled: bool = True
    destinations: List[LogDestination] = None
    custom_format: Optional[LogFormatConfig] = None
    
    def __post_init__(self):
        if self.destinations is None:
            self.destinations = [LogDestination.CONSOLE, LogDestination.FILE]


@dataclass
class RemoteLogConfig:
    """Remote logging configuration."""
    enabled: bool = False
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    batch_size: int = 100
    batch_timeout_seconds: int = 30


@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = False
    email_recipients: List[str] = None
    slack_webhook: Optional[str] = None
    error_threshold: int = 10
    critical_threshold: int = 5
    time_window_minutes: int = 15
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


@dataclass
class LoggingConfig:
    """Complete logging configuration."""
    environment: Environment
    log_directory: str
    global_log_level: str
    retention_policy: LogRetentionPolicy
    format_config: LogFormatConfig
    component_configs: Dict[str, ComponentLogConfig]
    remote_config: RemoteLogConfig
    alert_config: AlertConfig
    sensitive_data_patterns: List[str]
    
    def __post_init__(self):
        # Ensure log directory exists
        Path(self.log_directory).mkdir(parents=True, exist_ok=True)


class LoggingConfigManager:
    """Manages logging configuration for different environments."""
    
    def __init__(self, config_file: str = "logging_config.json"):
        self.config_file = Path(config_file)
        self.current_config: Optional[LoggingConfig] = None
        self._load_config()
    
    def _get_default_config(self, environment: Environment) -> LoggingConfig:
        """Get default configuration for environment."""
        
        # Base configuration
        base_config = {
            "environment": environment,
            "log_directory": "logs",
            "retention_policy": LogRetentionPolicy(),
            "format_config": LogFormatConfig(),
            "remote_config": RemoteLogConfig(),
            "alert_config": AlertConfig(),
            "sensitive_data_patterns": [
                r"password",
                r"token",
                r"api_key",
                r"secret",
                r"authorization",
                r"cookie",
                r"session_token",
                r"private_key",
                r"client_secret"
            ]
        }
        
        # Environment-specific configurations
        if environment == Environment.DEVELOPMENT:
            base_config.update({
                "global_log_level": "DEBUG",
                "component_configs": {
                    "api_server": ComponentLogConfig(
                        log_level="DEBUG",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "gemini_generator": ComponentLogConfig(
                        log_level="DEBUG",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "thumbnail_generator": ComponentLogConfig(
                        log_level="DEBUG",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "session_manager": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "progress_manager": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "file_validator": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "pipeline_stage": ComponentLogConfig(
                        log_level="DEBUG",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "frontend": ComponentLogConfig(
                        log_level="DEBUG",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    ),
                    "system": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.CONSOLE, LogDestination.FILE]
                    )
                }
            })
            
        elif environment == Environment.PRODUCTION:
            base_config.update({
                "global_log_level": "INFO",
                "format_config": LogFormatConfig(
                    include_timestamp=True,
                    include_session_id=True,
                    include_correlation_id=True,
                    include_memory_usage=False,
                    include_execution_time=True,
                    json_format=True,
                    pretty_print=False
                ),
                "retention_policy": LogRetentionPolicy(
                    max_file_size_mb=500,
                    max_files_per_component=20,
                    days_to_keep=90,
                    compress_old_logs=True
                ),
                "alert_config": AlertConfig(
                    enabled=True,
                    error_threshold=50,
                    critical_threshold=10,
                    time_window_minutes=30
                ),
                "component_configs": {
                    "api_server": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.FILE]
                    ),
                    "gemini_generator": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.FILE]
                    ),
                    "thumbnail_generator": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.FILE]
                    ),
                    "session_manager": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.FILE]
                    ),
                    "progress_manager": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.FILE]
                    ),
                    "file_validator": ComponentLogConfig(
                        log_level="WARNING",
                        destinations=[LogDestination.FILE]
                    ),
                    "pipeline_stage": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.FILE]
                    ),
                    "frontend": ComponentLogConfig(
                        log_level="WARNING",
                        destinations=[LogDestination.FILE]
                    ),
                    "system": ComponentLogConfig(
                        log_level="INFO",
                        destinations=[LogDestination.FILE]
                    )
                }
            })
            
        elif environment == Environment.TESTING:
            base_config.update({
                "global_log_level": "WARNING",
                "retention_policy": LogRetentionPolicy(
                    max_file_size_mb=10,
                    max_files_per_component=3,
                    days_to_keep=7,
                    compress_old_logs=False
                ),
                "component_configs": {
                    "api_server": ComponentLogConfig(
                        log_level="WARNING",
                        destinations=[LogDestination.FILE]
                    ),
                    "gemini_generator": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    ),
                    "thumbnail_generator": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    ),
                    "session_manager": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    ),
                    "progress_manager": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    ),
                    "file_validator": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    ),
                    "pipeline_stage": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    ),
                    "frontend": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    ),
                    "system": ComponentLogConfig(
                        log_level="ERROR",
                        destinations=[LogDestination.FILE]
                    )
                }
            })
        
        return LoggingConfig(**base_config)
    
    def _load_config(self):
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                    
                # Convert string enum values back to enum objects
                environment = Environment(config_data['environment'])
                
                # Reconstruct complex objects
                retention_policy = LogRetentionPolicy(**config_data['retention_policy'])
                format_config = LogFormatConfig(**config_data['format_config'])
                remote_config = RemoteLogConfig(**config_data['remote_config'])
                alert_config = AlertConfig(**config_data['alert_config'])
                
                # Reconstruct component configs
                component_configs = {}
                for component, config in config_data['component_configs'].items():
                    # Convert destination strings back to enums
                    destinations = [LogDestination(dest) for dest in config['destinations']]
                    component_configs[component] = ComponentLogConfig(
                        log_level=config['log_level'],
                        enabled=config['enabled'],
                        destinations=destinations,
                        custom_format=LogFormatConfig(**config['custom_format']) if config.get('custom_format') else None
                    )
                
                self.current_config = LoggingConfig(
                    environment=environment,
                    log_directory=config_data['log_directory'],
                    global_log_level=config_data['global_log_level'],
                    retention_policy=retention_policy,
                    format_config=format_config,
                    component_configs=component_configs,
                    remote_config=remote_config,
                    alert_config=alert_config,
                    sensitive_data_patterns=config_data['sensitive_data_patterns']
                )
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error loading config file: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration."""
        # Detect environment from environment variable
        env_name = os.getenv('PIPELINE_ENV', 'development')
        try:
            environment = Environment(env_name)
        except ValueError:
            environment = Environment.DEVELOPMENT
        
        self.current_config = self._get_default_config(environment)
        self.save_config()
    
    def save_config(self):
        """Save current configuration to file."""
        if not self.current_config:
            return
        
        # Convert to serializable format
        config_dict = asdict(self.current_config)
        
        # Convert enums to strings
        config_dict['environment'] = self.current_config.environment.value
        
        # Convert component config destinations
        for component, config in config_dict['component_configs'].items():
            config['destinations'] = [dest.value for dest in self.current_config.component_configs[component].destinations]
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_config(self) -> LoggingConfig:
        """Get current configuration."""
        return self.current_config
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        if not self.current_config:
            return
        
        for key, value in kwargs.items():
            if hasattr(self.current_config, key):
                setattr(self.current_config, key, value)
        
        self.save_config()
    
    def set_environment(self, environment: Environment):
        """Set environment and update configuration."""
        self.current_config = self._get_default_config(environment)
        self.save_config()
    
    def set_component_log_level(self, component: str, log_level: str):
        """Set log level for specific component."""
        if component in self.current_config.component_configs:
            self.current_config.component_configs[component].log_level = log_level
            self.save_config()
    
    def enable_component(self, component: str, enabled: bool = True):
        """Enable or disable logging for component."""
        if component in self.current_config.component_configs:
            self.current_config.component_configs[component].enabled = enabled
            self.save_config()
    
    def configure_remote_logging(self, endpoint: str, api_key: str, enabled: bool = True):
        """Configure remote logging."""
        self.current_config.remote_config.enabled = enabled
        self.current_config.remote_config.endpoint = endpoint
        self.current_config.remote_config.api_key = api_key
        self.save_config()
    
    def configure_alerts(self, email_recipients: List[str] = None, 
                        slack_webhook: str = None, enabled: bool = True):
        """Configure alerting."""
        self.current_config.alert_config.enabled = enabled
        if email_recipients:
            self.current_config.alert_config.email_recipients = email_recipients
        if slack_webhook:
            self.current_config.alert_config.slack_webhook = slack_webhook
        self.save_config()
    
    def get_component_config(self, component: str) -> Optional[ComponentLogConfig]:
        """Get configuration for specific component."""
        return self.current_config.component_configs.get(component)
    
    def is_component_enabled(self, component: str) -> bool:
        """Check if component logging is enabled."""
        config = self.get_component_config(component)
        return config.enabled if config else False
    
    def get_component_log_level(self, component: str) -> str:
        """Get log level for component."""
        config = self.get_component_config(component)
        return config.log_level if config else self.current_config.global_log_level
    
    def should_log_to_destination(self, component: str, destination: LogDestination) -> bool:
        """Check if component should log to specific destination."""
        config = self.get_component_config(component)
        return destination in config.destinations if config else False
    
    def get_log_file_path(self, component: str, date_str: str) -> Path:
        """Get log file path for component and date."""
        return Path(self.current_config.log_directory) / component / f"{component}_{date_str}.log"
    
    def cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        from datetime import datetime, timedelta
        
        retention = self.current_config.retention_policy
        cutoff_date = datetime.now() - timedelta(days=retention.days_to_keep)
        
        log_dir = Path(self.current_config.log_directory)
        for log_file in log_dir.rglob('*.log'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    if retention.compress_old_logs:
                        # Compress before deletion (implementation depends on requirements)
                        pass
                    log_file.unlink()
                except Exception as e:
                    print(f"Failed to delete old log file {log_file}: {e}")


# Global configuration manager
config_manager = LoggingConfigManager()


def get_config() -> LoggingConfig:
    """Get current logging configuration."""
    return config_manager.get_config()


def get_component_config(component: str) -> Optional[ComponentLogConfig]:
    """Get configuration for specific component."""
    return config_manager.get_component_config(component)


def is_component_enabled(component: str) -> bool:
    """Check if component logging is enabled."""
    return config_manager.is_component_enabled(component)


def get_component_log_level(component: str) -> str:
    """Get log level for component."""
    return config_manager.get_component_log_level(component)


def should_log_to_destination(component: str, destination: LogDestination) -> bool:
    """Check if component should log to specific destination."""
    return config_manager.should_log_to_destination(component, destination)


def set_environment(environment: Environment):
    """Set environment and update configuration."""
    config_manager.set_environment(environment)


def configure_remote_logging(endpoint: str, api_key: str, enabled: bool = True):
    """Configure remote logging."""
    config_manager.configure_remote_logging(endpoint, api_key, enabled)


def configure_alerts(email_recipients: List[str] = None, 
                    slack_webhook: str = None, enabled: bool = True):
    """Configure alerting."""
    config_manager.configure_alerts(email_recipients, slack_webhook, enabled)


if __name__ == "__main__":
    # Test configuration manager
    print("Testing logging configuration...")
    
    # Get current config
    config = get_config()
    print(f"Current environment: {config.environment.value}")
    print(f"Global log level: {config.global_log_level}")
    
    # Test component configuration
    api_config = get_component_config("api_server")
    print(f"API server log level: {api_config.log_level}")
    print(f"API server enabled: {api_config.enabled}")
    
    # Test environment switching
    set_environment(Environment.PRODUCTION)
    print(f"Switched to production environment")
    
    config = get_config()
    print(f"New global log level: {config.global_log_level}")
    
    print("Configuration test completed successfully!")