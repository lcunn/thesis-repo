import pandas as pd
import sqlite3
import musicbrainzngs
import os
import time

from typing import Optional, Dict, Any
from src.defaults import *
from scripts.utils.musicbrainz_utils import set_musicbrainz_api, search_musicbrainz
from scripts.database.utils.process_court_works import process_cases
from scripts.database.utils.process_mb_result import validate_mb_results

def _reformat_copyright_claims_csv(csv_path: str = COPYRIGHT_CLAIMS_CSV) -> pd.DataFrame:
    
    df = pd.read_csv(csv_path)
    complainers = df[['case_id', 'year', 'case_name', 'complaining_author', 'complaining_work']].rename(
        columns={'complaining_author': 'case_artist', 'complaining_work': 'case_title'})
    complainers['case_role'] = 'complainer'
    
    defendants = df[['case_id', 'year', 'case_name', 'defending', 'defending_work']].rename(
        columns={'defending': 'case_artist', 'defending_work': 'case_title'})
    defendants['case_role'] = 'defendant'

    songs = pd.concat([complainers, defendants], ignore_index=True)
    return songs

def _add_mb_data(df: pd.DataFrame, verbose: bool = True, filter: bool = True) -> pd.DataFrame:
    
    songs = df
    if filter:
        songs = songs[~(songs['gpt_artist'].isna() | songs['gpt_title'].isna())].copy(deep=True)
        
    t1 = time.time()
    songs['musicbrainz_data'] = songs.apply(
        lambda row: search_musicbrainz(row['gpt_artist'], row['gpt_title']), axis=1
    )
    t2 = time.time()
    print(f'Processing done in {(t2-t1)/60} mins') if verbose else None

    songs['musicbrainz_id'] = songs['musicbrainz_data'].apply(lambda x: x['id'] if x else None)
    songs['mb_artist'] = songs['musicbrainz_data'].apply(lambda x: x['artist-credit-phrase'] if x else None)
    songs['mb_title'] = songs['musicbrainz_data'].apply(lambda x: x['title'] if x else None)
    songs['case_status'] = None

    return songs.drop('musicbrainz_data', axis=1)

def process_copyright_songs() -> pd.DataFrame:
    """Process the copyright cases DataFrame and create a new DataFrame for songs."""
    
    set_musicbrainz_api()
    
    print('Processing copyright songs CSV...')
    if not os.path.exists(COPYRIGHT_SONGS_CSV):
        songs = _reformat_copyright_claims_csv()
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
        songs = _add_mb_data(songs)
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