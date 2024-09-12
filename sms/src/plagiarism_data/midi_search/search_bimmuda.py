import os
import pandas as pd
from fuzzywuzzy import fuzz
from typing import Dict, Union, Any, Protocol, List
from sms.defaults import *

from sms.src.plagiarism_data.midi_search.midi_searcher import MidiSearcher

class BiMMuDaSearcher(MidiSearcher):

    def set_search_path(self) -> None:
        self.source_name = 'bimmuda'
        self.directory = BIMMUDA_PATH

    def find_song_matches(self, artist: str, title: str) -> List[str]:

        metadata_df = pd.read_csv(BIMMUDA_METADATA_PATH)
        matches = []
        potential_matches = metadata_df[
            (metadata_df['Artist'].apply(lambda x: self.fuzzy_match(x, artist))) &
            (metadata_df['Title'].apply(lambda x: self.fuzzy_match(x, title)))
        ]

        for i, row in potential_matches.iterrows():
            year = str(row['Year'])
            position = str(row['Position'])

            base_path = os.path.join(self.directory, year, position)
            full_path = os.path.join(base_path, f"{year}_0{position}_full.mid")
            melody_path = os.path.join(base_path, f"{year}_0{position}_1.mid")

            if os.path.exists(full_path):
                matches.append(full_path)
            elif os.path.exists(melody_path):
                matches.append(melody_path)
            else:
                self.logger.warning(f"No MIDI file found for match {i+1}/{len(potential_matches)} for {artist} - {title}")

        return matches

    @staticmethod
    def fuzzy_match(s1: str, s2: str, threshold: int = 80) -> bool:
        return fuzz.ratio(s1.lower(), s2.lower()) > threshold

if __name__ == '__main__':
    bimmuda_searcher = BiMMuDaSearcher()
    bimmuda_searcher.search_directory()

