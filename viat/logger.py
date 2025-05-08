"""
Logger module for VIAT application.

Provides logging functionality to record application events and crashes.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from functools import wraps

class VIATLogger:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VIATLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir=None):
        if self._initialized:
            return
            
        self.logger = logging.getLogger('VIAT')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Determine log directory
        if log_dir is None:
            # Use the current directory as fallback
            self.log_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.log_dir = log_dir
            
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create log file
        self.log_file = os.path.join(self.log_dir, 'viat_errors.log')
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.ERROR)  # Only log errors and above
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        # Set up global exception handler
        self.setup_exception_handler()
        
        self._initialized = True
        
    def setup_exception_handler(self):
        """Set up global exception handler to catch unhandled exceptions."""
        def exception_hook(exctype, value, tb):
            # Log the exception
            exception_str = ''.join(traceback.format_exception(exctype, value, tb))
            self.logger.critical(f"UNHANDLED EXCEPTION: {exception_str}")
            # Call the original exception hook
            sys.__excepthook__(exctype, value, tb)
        
        # Set the exception hook
        sys.excepthook = exception_hook
    
    def log_error(self, error, context=""):
        """Log an error with context information."""
        if isinstance(error, Exception):
            tb_str = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
            self.logger.error(f"ERROR in {context}: {str(error)}\n{tb_str}")
        else:
            self.logger.error(f"ERROR in {context}: {str(error)}")

# Create a decorator for logging exceptions in functions
def log_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the exception
            logger = logging.getLogger(__name__)
            logger.exception(f"Exception in {func.__name__}: {str(e)}")
            # You can add additional error handling here if needed
            raise  # Re-raise the exception after logging
    return wrapper