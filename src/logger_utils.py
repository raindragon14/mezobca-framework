import logging
import os
import sys
from datetime import datetime

def setup_logger(name: str = "MeZO-BCA", log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Setup a standardized logger that outputs to both console and a log file.
    
    Args:
        name: Name of the logger.
        log_dir: Directory where log files are stored.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # Setup formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f"{name.lower()}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging in directory {log_dir}: {e}")

    return logger

# Create a default logger instance
logger = setup_logger()
