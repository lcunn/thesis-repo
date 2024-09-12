import os
import pandas as pd
import json
from fuzzywuzzy import fuzz
from typing import Dict, Union, Any, Protocol, List
from sms.defaults import *

from sms.src.plagiarism_data.midi_search.midi_searcher import MidiSearcher

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

class MetaMIDISearcher(MidiSearcher):

    def set_search_path(self) -> None:
        self.source_name = 'metamidi'
        self.directory = METAMIDI_PATH
        self.directory_list = self._get_directory_list()
        self.metadata_df = self._get_metadata()

    def _get_directory_list(self) -> List[str]:
        directory_list = []
        for root, _, files in os.walk(METAMIDI_PATH):
            for file in files:
                full_path = os.path.join(root, file)
                directory_list.append(full_path)
        return directory_list
    
    def _get_metadata(self) -> pd.DataFrame:
        metamidi_metadata = read_jsonl(METAMIDI_METADATA_PATH)
        df = pd.DataFrame([
            {
                'md5': entry['md5'],
                'title': entry['title_artist'][0][0],
                'artist': entry['title_artist'][0][1]
            }
            for entry in metamidi_metadata
        ])
        return df

    def find_song_matches(self, artist: str, title: str) -> List[str]:

        matches = []
        potential_matches = self.metadata_df[
            (self.metadata_df['artist'].apply(lambda x: self.fuzzy_match(x, artist))) &
            (self.metadata_df['title'].apply(lambda x: self.fuzzy_match(x, title)))
        ]

        for i, row in potential_matches.iterrows():
            id = row['md5']
            matching_files = [file for file in self.directory_list if id in file and file.endswith('.mid')]
            if matching_files:
                matches.extend(matching_files)
            else:
                self.logger.warning(f"No MIDI file found for match {i+1}/{len(potential_matches)} for {artist} - {title}")

        return matches

    @staticmethod
    def fuzzy_match(s1: str, s2: str, threshold: int = 80) -> bool:
        return fuzz.ratio(s1.lower(), s2.lower()) > threshold

if __name__ == '__main__':
    metamidi_searcher = MetaMIDISearcher()
    metamidi_searcher.search_directory()

