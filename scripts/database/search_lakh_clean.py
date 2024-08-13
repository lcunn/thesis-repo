import os
import shutil
import logging
import pandas as pd
from fuzzywuzzy import fuzz
from typing import Dict, Union, Any, Protocol, List
from src.defaults import *

from scripts.database.midi_searcher import MidiSearcher

def lakh_clean_compare(dir_list: List[str], artist: str, title: str) -> List[str]:
    """
    lakh_clean directory is formatted as artist/song. 
    Performs fuzzy matching to account for slight inaccuracies in artist/song names.
    """
    
    matches = []
    for path in dir_list:
        # Extract artist and song from the path
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) < 2:
            continue
        path_artist, path_song = parts[-2], os.path.splitext(parts[-1])[0]
        
        # Perform fuzzy matching
        artist_ratio = fuzz.ratio(artist.lower(), path_artist.lower())
        title_ratio = fuzz.ratio(title.lower(), path_song.lower())
        
        # If both artist and title have a high match ratio, consider it a match
        if artist_ratio > 80 and title_ratio > 80:
            matches.append(path)
    
    return matches

config = {
    'lakh_clean': lakh_clean_compare
}

if __name__ == '__main__':
    LakhCleanSearcher = MidiSearcher(config)
    LakhCleanSearcher.search_directory('lakh_clean')

