import pandas as pd
import time

from src.defaults import *
from scripts.utils.musicbrainz_utils import set_musicbrainz_api, search_musicbrainz

def add_mb_data(df: pd.DataFrame, verbose: bool = True, filter: bool = True) -> pd.DataFrame:

    set_musicbrainz_api()
    
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