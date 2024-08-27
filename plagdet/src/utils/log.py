import logging

from typing import Optional

from plagdet.src.defaults import *

def configure_logging(
    level: int = logging.INFO,
    to_console: bool = True,
    to_file: bool = False,
    log_file: Optional[str] = None,
    format: str = '[%(asctime)s] [%(levelname)-5s] %(message)s',
    datefmt: str = '%Y-%m-%d %H:%M:%S'
):
    handlers = []
    
    if to_console:
        console_handler = logging.StreamHandler()
        handlers.append(console_handler)
    
    if to_file and log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format=format,
        datefmt=datefmt,
        handlers=handlers
    )