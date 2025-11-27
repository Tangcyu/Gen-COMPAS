# utils/logger.py
import logging
import sys 

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """ Creates and configures a logger. """
    logger = logging.getLogger(name)
    if not logger.handlers: 
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
