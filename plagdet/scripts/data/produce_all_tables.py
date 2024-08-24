import pandas as pd
import os
import pickle as pkl
import logging

from typing import Optional, Dict, Any, Callable

from plagdet.src.defaults import *

from plagdet.scripts.data.helper.retrieve_copyright_claims import produce_table

from plagdet.scripts.data.filter_cases.filter_claims import filter_unwanted_cases

from plagdet.scripts.data.process_cases.process_cases_with_ai import estimate_songs_from_cases

from plagdet.scripts.data.final_formatting.format_pairs import process_copyright_pairs

# from plagdet.scripts.csvs.helper.add_mb_data import add_mb_data
# from plagdet.scripts.csvs.helper.process_mb_result import validate_mb_results

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)-5s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def conduct_step(
        message: str, 
        func: Callable[[Any], pd.DataFrame],
        dest: str
        ) -> pd.DataFrame:
    
    dest_is_pkl = dest.endswith('.pkl')
    logger.info(message)
    if not os.path.exists(dest):
        df = func()
        if dest_is_pkl:
            df.to_pickle(dest)
        else:
            df.to_csv(dest, index=False)
    else:
        logger.info('Already processed.')
        if dest_is_pkl:
            df = pd.read_pickle(dest)
        else:
            df = pd.read_csv(dest)
    return dest

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

    if not os.path.exists(VALIDATED_SONGS_CSV):
        logger.error(f"The file {VALIDATED_SONGS_CSV} does not exist. Please ensure the validated songs pickle file has been created and is in the correct location.")
        raise FileNotFoundError(f"Missing file: {VALIDATED_SONGS_CSV}")

    # conduct_step(
    #     'Formatting validated copyright song pairs into a CSV...',
    #     process_copyright_pairs,
    #     COPYRIGHT_SONGS_CSV_GPT
    # )

    # print('\nAdding mb data...')
    # if not os.path.exists(COPYRIGHT_SONGS_CSV_GPT_MB):
    #     songs = add_mb_data(songs)
    #     songs.to_csv(COPYRIGHT_SONGS_CSV_GPT_MB, index=False)
    # else:
    #     print('Already processed.')
    #     songs = pd.read_csv(COPYRIGHT_SONGS_CSV_GPT_MB)

    # logger.info('Processing copyright songs CSV...')
    # if not os.path.exists(COPYRIGHT_SONGS_CSV):
    #     songs = reformat_copyright_claims()
    #     songs.to_csv(COPYRIGHT_SONGS_CSV, index=False)
    # else:
    #     print('Already processed.')
    #     songs = pd.read_csv(COPYRIGHT_SONGS_CSV)
    
    # print('\nProcessing cases into title and artist...')
    # if not os.path.exists(COPYRIGHT_SONGS_CSV_GPT):
    #     songs = process_cases(songs)
    #     songs.to_csv(COPYRIGHT_SONGS_CSV_GPT, index=False)
    # else:
    #     print('Already processed.')
    #     songs = pd.read_csv(COPYRIGHT_SONGS_CSV_GPT)

    # print('\nAdding mb data...')
    # if not os.path.exists(COPYRIGHT_SONGS_CSV_GPT_MB):
    #     songs = add_mb_data(songs)
    #     songs.to_csv(COPYRIGHT_SONGS_CSV_GPT_MB, index=False)
    # else:
    #     print('Already processed.')
    #     songs = pd.read_csv(COPYRIGHT_SONGS_CSV_GPT_MB)

    # print('\nValidating mb data...')
    # if not os.path.exists(COPYRIGHT_SONGS_CSV_GPT_MB_V):
    #     songs = validate_mb_results(songs)
    #     songs.to_csv(COPYRIGHT_SONGS_CSV_GPT_MB_V, index=False)
    # else:
    #     print('Already processed.')
    #     songs = pd.read_csv(COPYRIGHT_SONGS_CSV_GPT_MB_V)
    
    # return songs

if __name__ == "__main__":
    process_copyright_songs()