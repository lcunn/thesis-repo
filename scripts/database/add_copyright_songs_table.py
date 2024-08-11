import pandas as pd
import sqlite3
import musicbrainzngs
import os

from typing import Optional, Dict, Any
from src.defaults import *

def set_musicbrainz_api() -> None:
    """Set musicbrainz api agent."""
    musicbrainzngs.set_useragent(*list(MUSICBRAINZ_AGENT.values()))

def search_musicbrainz(artist: str, title: str) -> Optional[Dict[str, Any]]:
    """Search musicbrainz for a song by artist and title."""
    result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
    if result['recording-list']:
        return result['recording-list'][0]
    return None

def process_copyright_songs(df: pd.DataFrame) -> pd.DataFrame:
    """Process the copyright cases DataFrame and create a new DataFrame for songs."""
    
    set_musicbrainz_api()

    df = df.reset_index().copy()

    complainers = df[['index', 'complaining_author', 'complaining_work']].rename(
        columns={'complaining_author': 'case_artist', 'complaining_work': 'case_title'})
    complainers['case_role'] = 'complainer'
    
    defendants = df[['index', 'defending', 'defending_work']].rename(
        columns={'defending': 'case_artist', 'defending_work': 'case_title'})
    defendants['case_role'] = 'defendant'
    
    # Combine the DataFrames
    songs = pd.concat([complainers, defendants], ignore_index=True)
    
    # Apply musicbrainz search to all rows at once
    songs['musicbrainz_data'] = songs.apply(
        lambda row: search_musicbrainz(row['case_artist'], row['case_title']), axis=1
    )
    
    songs['musicbrainz_id'] = songs['musicbrainz_data'].apply(lambda x: x['id'] if x else None)
    songs['mb_artist'] = songs['musicbrainz_data'].apply(lambda x: x['artist-credit-phrase'] if x else None)
    songs['mb_title'] = songs['musicbrainz_data'].apply(lambda x: x['title'] if x else None)
    songs['case_status'] = None

    songs = (songs
             .drop('musicbrainz_data', axis=1)
             .rename(columns={'index': 'case_id'})
             .set_index('musicbrainz_id')
             )
    
    return songs[['case_artist', 'case_title', 'case_role', 'case_status', 
                  'mb_artist', 'mb_title', 'case_id']]

def add_songs_to_table(songs_df: pd.DataFrame) -> None:
    """Adds processed songs DataFrame to the database."""
    conn = sqlite3.connect(DATABASE)
    
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS copyright_songs (
        musicbrainz_id TEXT PRIMARY KEY,
        case_artist TEXT,
        case_title TEXT,
        case_id INTEGER,
        case_role TEXT,
        case_status BOOL,
        mb_artist TEXT,
        mb_title TEXT,
        midi_file_path TEXT,
        FOREIGN KEY (case_id) REFERENCES copyright_cases(id)
    );
    '''
    conn.execute(create_table_query)
    songs_df.to_sql('copyright_songs', conn, if_exists='append', 
                    index=True, index_label='musicbrainz_id')
    conn.commit()
    conn.close()

if __name__ == '__main__':    
    print(f"Reading processed songs from {COPYRIGHT_SONGS_CSV}")
    songs_df = pd.read_csv(COPYRIGHT_SONGS_CSV, index_col='musicbrainz_id')
    # Add the songs to the database table
    print("Adding songs to the database table")
    add_songs_to_table(songs_df)
    
    print("Process completed successfully")