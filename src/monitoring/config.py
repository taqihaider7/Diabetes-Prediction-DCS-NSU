"""
Monitoring Configuration
Central configuration for logging and metrics
"""

import os
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env in project root (one level up from src)
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip loading
    pass


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""
    
    # Loki configuration (Grafana Cloud or Self-hosted)
    loki_enabled: bool = True
    loki_url: str = "http://localhost:3100/loki/api/v1/push"  # Or Grafana Cloud URL
    loki_username: Optional[str] = None  # Grafana Cloud instance ID
    loki_api_key: Optional[str] = None   # Grafana Cloud API key
    
    # Application settings
    app_name: str = "diabetes-prediction"
    environment: str = "development"  # development, staging, production
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "../logs"
    
    # Metrics settings
    metrics_retention_hours: int = 24
    metrics_export_enabled: bool = True
    metrics_dir: str = "../metrics"
    
    # Session tracking
    track_user_sessions: bool = True
    
    @classmethod
    def from_env(cls) -> 'MonitoringConfig':
        """
        Load configuration from environment variables
        
        Returns:
            MonitoringConfig instance
        """
        return cls(
            loki_enabled=os.getenv('LOKI_ENABLED', 'true').lower() == 'true',
            loki_url=os.getenv('LOKI_URL', 'http://localhost:3100/loki/api/v1/push'),
            loki_username=os.getenv('LOKI_USERNAME'),  # Grafana Cloud username
            loki_api_key=os.getenv('LOKI_API_KEY'),    # Grafana Cloud API key
            app_name=os.getenv('APP_NAME', 'diabetes-prediction'),
            environment=os.getenv('ENVIRONMENT', 'development'),
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            log_to_file=os.getenv('LOG_TO_FILE', 'true').lower() == 'true',
            log_dir=os.getenv('LOG_DIR', '../logs'),
            metrics_retention_hours=int(os.getenv('METRICS_RETENTION_HOURS', '24')),
            metrics_export_enabled=os.getenv('METRICS_EXPORT_ENABLED', 'true').lower() == 'true',
            metrics_dir=os.getenv('METRICS_DIR', '../metrics'),
            track_user_sessions=os.getenv('TRACK_USER_SESSIONS', 'true').lower() == 'true'
        )


# Default configuration
DEFAULT_CONFIG = MonitoringConfig.from_env()
