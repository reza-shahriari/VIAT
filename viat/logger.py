"""
Logging utilities for the Video Annotation Tool.
"""

import os
import sys
import logging
import traceback
from functools import wraps
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# Configure the logger
log_file = os.path.join(logs_dir, f'viat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

class VIATLogger:
    """Logger class for the Video Annotation Tool."""
    
    def __init__(self):
        self.logger = logging.getLogger('VIAT')
        print('Logging into: ',logs_dir)    
    def info(self, message):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error message."""
        self.logger.error(message)
    
    def exception(self, message):
        """Log an exception with traceback."""
        self.logger.exception(message)

# Create a simpler decorator that won't break function calls
def log_exceptions(func):
    """
    A decorator that logs exceptions raised by the decorated function.
    This version handles both regular functions and class methods properly.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            # Check if this is a "takes X positional arguments but Y were given" error
            error_msg = str(e)
            if "positional argument" in error_msg and "were given" in error_msg:
                # Try to call with just the first argument (self) if it's a method
                if len(args) > 0:
                    try:
                        return func(args[0])
                    except Exception as inner_e:
                        logger = logging.getLogger('VIAT')
                        logger.error(f"Exception in {func.__name__}: {str(inner_e)}")
                        logger.error(traceback.format_exc())
                        raise
            # If not a positional argument error or the retry failed, log and re-raise
            logger = logging.getLogger('VIAT')
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        except Exception as e:
            logger = logging.getLogger('VIAT')
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            # Re-raise the exception to maintain original behavior
            raise
    return wrapper
