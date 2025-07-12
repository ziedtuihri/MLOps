import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging(log_file='mlflow.log', level=logging.INFO):
    """
    Setup logging configuration for the application.
    This creates a log file that can be consumed by Logstash.
    
    Args:
        log_file (str): Path to the log file
        level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            # File handler for Logstash consumption
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            # Console handler for immediate feedback
            logging.StreamHandler()
        ]
    )
    
    # Create a logger for this application
    logger = logging.getLogger('mlflow_app')
    logger.setLevel(level)
    
    return logger

def get_logger(name='mlflow_app'):
    """
    Get a logger instance.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name) 