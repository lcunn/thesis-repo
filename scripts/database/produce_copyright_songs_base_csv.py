import pandas as pd
import sqlite3
import musicbrainzngs
import os
import time

from typing import Optional, Dict, Any
from src.defaults import *
from scripts.utils.musicbrainz_utils import set_musicbrainz_api, search_musicbrainz

def _reformat_songs_df(df: pd.DataFrame) -> pd.DataFrame:
        
    complainers = df[['case_id', 'year', 'case_name', 'complaining_author', 'complaining_work']].rename(
        columns={'complaining_author': 'case_artist', 'complaining_work': 'case_title'})
    complainers['case_role'] = 'complainer'
    
    defendants = df[['case_id', 'year', 'case_name', 'defending', 'defending_work']].rename(
        columns={'defending': 'case_artist', 'defending_work': 'case_title'})
    defendants['case_role'] = 'defendant'

    songs = pd.concat([complainers, defendants], ignore_index=True)
    return songs
    
def _add_mb_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    
    songs = df
    print('Doing musicbrainz processing') if verbose else None
    t1 = time.time()
    songs['musicbrainz_data'] = songs.apply(
        lambda row: search_musicbrainz(row['case_artist'], row['case_title']), axis=1
    )
    t2 = time.time()
    print(f'Processing done in {(t2-t1)/60} mins') if verbose else None

    songs['musicbrainz_id'] = songs['musicbrainz_data'].apply(lambda x: x['id'] if x else None)
    songs['mb_artist'] = songs['musicbrainz_data'].apply(lambda x: x['artist-credit-phrase'] if x else None)
    songs['mb_title'] = songs['musicbrainz_data'].apply(lambda x: x['title'] if x else None)
    songs['case_status'] = None

    return songs.drop('musicbrainz_data', axis=1)


def process_copyright_songs(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Process the copyright cases DataFrame and create a new DataFrame for songs."""
    
    set_musicbrainz_api()

    songs = _reformat_songs_df(df)
    songs = _add_mb_data(songs)
    
    return songs[['musicbrainz_id', 'case_id', 'case_artist', 'case_title', 'case_role', 'case_status', 
                  'mb_artist', 'mb_title']]

if __name__ == "__main__":
    if os.path.exists(COPYRIGHT_SONGS_CSV):
        # If the processed songs CSV exists, quit
        print(f"{COPYRIGHT_SONGS_CSV} already exists.")
    else:
        print(f"Processing copyright cases from {COPYRIGHT_CLAIMS_CSV}")
        cases_df = pd.read_csv(COPYRIGHT_CLAIMS_CSV)
        songs_df = process_copyright_songs(cases_df)
        print('Writing to CSV')
        songs_df.to_csv(COPYRIGHT_SONGS_CSV)