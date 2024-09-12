import logging

from typing import Optional

from sms.defaults import *

def configure_logging(
    console_level: int = logging.DEBUG,
    file_level: int = logging.INFO,
    to_console: bool = True,
    to_file: bool = False,
    log_file: Optional[str] = None,
    format: str = '[%(asctime)s] [%(levelname)-5s] %(message)s',
    datefmt: str = '%Y-%m-%d %H:%M:%S'
):
    handlers = []
    
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        handlers.append(console_handler)
    
    if to_file and log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=min(console_level, file_level),
        format=format,
        datefmt=datefmt,
        handlers=handlers
    )