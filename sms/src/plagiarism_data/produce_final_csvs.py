import pandas as pd
import os
import pickle as pkl
import logging

from typing import Optional, Dict, Any, Callable

from sms.defaults import *

from sms.src.plagiarism_data.final_formatting.format_pairs import process_copyright_pairs

def process_copyright_songs() -> pd.DataFrame:
    """Process the copyright cases DataFrame and create a new DataFrame for songs."""

    songs_exist = os.path.exists(COPYRIGHT_SONGS_CSV)
    pairs_exist = os.path.exists(COPYRIGHT_PAIRS_CSV)

    if not songs_exist and not pairs_exist:
        songs, pairs = process_copyright_pairs()
        songs.to_csv(COPYRIGHT_SONGS_CSV, index=False)
        pairs.to_csv(COPYRIGHT_PAIRS_CSV, index=False)
    elif not songs_exist and pairs_exist:
        songs, _ = process_copyright_pairs()
        songs.to_csv(COPYRIGHT_SONGS_CSV, index=False)
    elif songs_exist and not pairs_exist:
        _, pairs = process_copyright_pairs()
        pairs.to_csv(COPYRIGHT_PAIRS_CSV, index=False)
    else:
        print("Both CSV files already exist. Skipping processing.")


if __name__ == "__main__":
    process_copyright_songs()