"""
Logging Configuration with Grafana Loki Integration
Uses official python-logging-loki library for reliable log shipping
"""

import logging
import logging.handlers
import sys
import os
from typing import Optional
from pathlib import Path
from datetime import datetime

try:
    import logging_loki
    LOKI_AVAILABLE = True
except ImportError:
    LOKI_AVAILABLE = False
    print("Warning: python-logging-loki not installed. Install with: pip install python-logging-loki")


def setup_loki_logging(
    app_name: str = "diabetes-prediction",
    loki_url: Optional[str] = None,
    log_level: int = logging.INFO,
    environment: str = "development",
    loki_username: Optional[str] = None,
    loki_api_key: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging with Grafana Loki integration using python-logging-loki library
    
    Args:
        app_name: Application name for labeling
        loki_url: Loki push API URL (if None, uses console only)
        log_level: Logging level
        environment: Environment name (dev, staging, prod)
        loki_username: Grafana Cloud username (instance ID)
        loki_api_key: Grafana Cloud API key
        
    Returns:
        Configured logger instance
    
    Example:
        logger = setup_loki_logging(
            app_name="my-app",
            loki_url="https://logs-prod-012.grafana.net/loki/api/v1/push",
            loki_username="123456",
            loki_api_key="glc_xxx..."
        )
        logger.info("Application started")
    """
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for local logs
    try:
        log_dir = Path(__file__).parent.parent.parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f'{app_name}.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.warning(f"Failed to setup file handler: {str(e)}")
    
    # Loki handler if URL provided and library available
    if loki_url and LOKI_AVAILABLE:
        try:
            # Prepare authentication for Grafana Cloud
            auth = None
            if loki_username and loki_api_key:
                auth = (loki_username, loki_api_key)
            
            # Create Loki handler with QueueHandler for async logging
            # Using QueueHandler + QueueListener pattern
            from queue import Queue
            
            # Create the base Loki handler
            loki_handler = logging_loki.LokiHandler(
                url=loki_url,
                tags={
                    "application": app_name,
                    "environment": environment,
                    "service": "streamlit"
                },
                auth=auth,
                version="1"  # Loki API version
            )
            
            # Wrap it in QueueHandler for async processing
            log_queue = Queue(-1)  # No size limit
            queue_handler = logging.handlers.QueueHandler(log_queue)
            queue_handler.setLevel(log_level)
            
            # Start the listener in a separate thread
            listener = logging.handlers.QueueListener(log_queue, loki_handler, respect_handler_level=True)
            listener.start()
            
            logger.addHandler(queue_handler)
            
            cloud_status = "Grafana Cloud" if loki_username else "Self-hosted"
            logger.info(f"✓ Loki logging enabled ({cloud_status})")
            
        except Exception as e:
            logger.warning(f"Failed to setup Loki handler: {str(e)}")
            logger.info("Application will continue with console/file logging only")
            
    elif loki_url and not LOKI_AVAILABLE:
        logger.warning("Loki URL provided but python-logging-loki not installed")
        logger.info("Install with: pip install python-logging-loki")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing data")
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding extra fields to logs
    
    Example:
        with LogContext(logger, prediction_id="abc123", user="john") as ctx:
            ctx.info("Starting prediction")
            # ... do work ...
            ctx.info("Prediction complete")
    """
    
    def __init__(self, logger: logging.Logger, **kwargs):
        """
        Initialize log context
        
        Args:
            logger: Logger instance
            **kwargs: Extra fields to add to all logs in this context
        """
        self.logger = logger
        self.extra = kwargs
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                f"Exception in context: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={"tags": self.extra}
            )
        return False
        
    def info(self, msg: str, **kwargs):
        """Log info message with context"""
        extra_tags = {**self.extra, **kwargs}
        self.logger.info(msg, extra={"tags": extra_tags})
        
    def error(self, msg: str, **kwargs):
        """Log error message with context"""
        extra_tags = {**self.extra, **kwargs}
        self.logger.error(msg, extra={"tags": extra_tags})
        
    def warning(self, msg: str, **kwargs):
        """Log warning message with context"""
        extra_tags = {**self.extra, **kwargs}
        self.logger.warning(msg, extra={"tags": extra_tags})
        
    def debug(self, msg: str, **kwargs):
        """Log debug message with context"""
        extra_tags = {**self.extra, **kwargs}
        self.logger.debug(msg, extra={"tags": extra_tags})
