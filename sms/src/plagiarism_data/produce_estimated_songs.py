import pandas as pd
import os
import pickle as pkl
import logging

from typing import Optional, Dict, Any, Callable

from sms.defaults import *
from sms.src.log import get_logger

from sms.src.plagiarism_data.retrieve_claims.retrieve_claims import produce_table

from sms.src.plagiarism_data.filter_cases.filter_claims import filter_unwanted_cases

from sms.src.plagiarism_data.process_cases.process_cases_with_ai import estimate_songs_from_cases

def conduct_step(
        message: str, 
        func: Callable[[Any], pd.DataFrame],
        dest: str,
        logger: logging.Logger = get_logger(__name__)
        ) -> None:
    
    dest_is_pkl = dest.endswith('.pkl')
    logger.info(message)
    if not os.path.exists(dest):
        result = func()
        if dest_is_pkl:
            result.to_pickle(dest)
        else:
            result.to_csv(dest, index=False)
    else:
        logger.info('Already processed.')
        if dest_is_pkl:
            result = pd.read_pickle(dest)
        else:
            result = pd.read_csv(dest)

def process_copyright_songs() -> pd.DataFrame:
    """Process the copyright cases DataFrame and create a new DataFrame for songs."""

    if not os.path.exists(COPYRIGHT_CLAIMS_CSV):
        produce_table()

    conduct_step(
        'Filtering unwanted cases...',
        filter_unwanted_cases,
        COPYRIGHT_CLAIMS_CSV_F
    )

    conduct_step(
        'Applying GPT to estimate songs from cases...',
        estimate_songs_from_cases,
        ESTIMATED_SONGS_CSV
    )

if __name__ == "__main__":
    process_copyright_songs()