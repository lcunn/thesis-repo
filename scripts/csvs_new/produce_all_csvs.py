import pandas as pd
import os

from typing import Optional, Dict, Any

from src.defaults import *

from scripts.csvs_new.retrieve_copyright_claims import produce_table

from scripts.csvs.helper.filter_claims import reformat_copyright_claims
from scripts.csvs.helper.process_cases_with_ai import process_cases
from scripts.csvs.helper.add_song_ids import add_song_ids
from scripts.csvs.helper.add_mb_data import add_mb_data
from scripts.csvs.helper.process_mb_result import validate_mb_results

def produce_csvs() -> pd.DataFrame:
    """Process the copyright cases DataFrame and create a new DataFrame for songs."""

    if not os.path.exists(COPYRIGHT_CLAIMS_CSV):
        produce_table()

    print('Processing copyright songs CSV...')
    if not os.path.exists(COPYRIGHT_SONGS_CSV):
        songs = reformat_copyright_claims()
        songs.to_csv(COPYRIGHT_SONGS_CSV, index=False)
    else:
        print('Already processed.')
        songs = pd.read_csv(COPYRIGHT_SONGS_CSV)
    
    print('\nProcessing cases into title and artist...')
    if not os.path.exists(COPYRIGHT_SONGS_CSV_GPT):
        songs = process_cases(songs)
        songs.to_csv(COPYRIGHT_SONGS_CSV_GPT, index=False)
    else:
        print('Already processed.')
        songs = pd.read_csv(COPYRIGHT_SONGS_CSV_GPT)

    print('\nAdding mb data...')
    if not os.path.exists(COPYRIGHT_SONGS_CSV_GPT_MB):
        songs = add_mb_data(songs)
        songs.to_csv(COPYRIGHT_SONGS_CSV_GPT_MB, index=False)
    else:
        print('Already processed.')
        songs = pd.read_csv(COPYRIGHT_SONGS_CSV_GPT_MB)

    print('\nValidating mb data...')
    if not os.path.exists(COPYRIGHT_SONGS_CSV_GPT_MB_V):
        songs = validate_mb_results(songs)
        songs.to_csv(COPYRIGHT_SONGS_CSV_GPT_MB_V, index=False)
    else:
        print('Already processed.')
        songs = pd.read_csv(COPYRIGHT_SONGS_CSV_GPT_MB_V)
    
    return songs

if __name__ == "__main__":
    process_copyright_songs()