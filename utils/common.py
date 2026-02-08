import os
import logging
import re

def setup_logger(name, log_file, level=logging.INFO):
    """
    Creates a logger that writes to a specific file.
    """
    # מוודא שהתיקייה של הלוג קיימת
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')        
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # מונע כפילות לוגים אם מריצים את הפונקציה שוב
    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def sanitize_filename(name):
    """
    Cleans file names from forbidden characters.
    """
    # משאיר רק אותיות, מספרים, רווחים, מקפים וקווים תחתונים
    clean_name = re.sub(r'[^\w\s-]', '', name)
    clean_name = re.sub(r'\s+', '_', clean_name)
    return clean_name.strip()