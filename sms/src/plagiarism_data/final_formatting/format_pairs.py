from typing import List, Optional, Tuple
import logging
import os

from openai import OpenAI
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz

from sms.defaults import *

# the validated songs dataframe has the following columns:
# 'validated'
# 'case_id'
# 'song1_artist'
# 'song1_title'
# 'song1_evidence'
# 'song2_artist'
# 'song2_title'
# 'song2_evidence'
# 'pair_evidence'
# 'is_melodic_comparison'
# 'melodic_evidence'
# 'was_case_won'
# 'case_won_evidence'

def check_song_id(artist: str, title: str, songs: pd.DataFrame, threshold: int = 90) -> Optional[int]:
    """
    Check if a song is already stored in the songs dataframe.
    If it is, return the song_id of the song.
    If it is not, return None.
    """
    for _, row in songs.iterrows():
        if (fuzz.ratio(artist.lower(), row['artist'].lower()) >= threshold and
            fuzz.ratio(title.lower(), row['title'].lower()) >= threshold):
            return row['song_id']
    return None

def process_copyright_pairs(file_path: str = VALIDATED_SONGS_CSV) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    pairs = pd.read_csv(file_path)

    copyright_songs = pd.DataFrame(columns=['song_id', 'artist', 'title'])
    copyright_cases = pd.DataFrame(columns=['case_id', 'complaining_id', 'defending_id', 'is_melodic_comparison', 'case_won'])

    song_id = 0

    for _, case in pairs.iterrows():
        song_ids = []
        for prefix in ['song1', 'song2']:
            # check if the song is already stored in the songs dataframe
            song_id_existing = check_song_id(case[f'{prefix}_artist'], case[f'{prefix}_title'], copyright_songs)
            # if the song is not stored, store it
            if song_id_existing is None:
                song_ids.append(song_id)
                new_song = pd.DataFrame({
                    'song_id': [song_id],
                    'artist': [case[f'{prefix}_artist']],
                    'title': [case[f'{prefix}_title']]
                })
                copyright_songs = pd.concat([copyright_songs, new_song], ignore_index=True)
                song_id += 1
            else:
                song_ids.append(song_id_existing)

        new_case = pd.DataFrame({
            'case_id': [case['case_id']],
            'complaining_id': [song_ids[0]],
            'defending_id': [song_ids[1]],
            'is_melodic_comparison': [case['is_melodic_comparison']],
            'case_won': [case['was_case_won']]
        })
        copyright_cases = pd.concat([copyright_cases, new_case], ignore_index=True)

    return copyright_songs, copyright_cases

            


