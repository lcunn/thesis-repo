import os
import pandas as pd
from fuzzywuzzy import fuzz
from typing import Dict, Union, Any, Protocol, List
from sms.defaults import *

from sms.src.plagiarism_data.midi_search.midi_searcher import MidiSearcher

class LakhCleanSearcher(MidiSearcher):

    def set_search_path(self) -> None:
        self.source_name = 'lakh_clean'
        self.directory = LAKH_CLEAN_PATH
        self.directory_list = self._make_directory_list()

    def _make_directory_list(self) -> List[str]:
        filepaths = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                filepaths.append(os.path.join(root, file))
        return filepaths
    
    def find_song_matches(self, artist: str, title: str) -> List[str]:
        matches = []
        for path in self.directory_list:
            parts = os.path.normpath(path).split(os.sep)
            if len(parts) < 2:
                continue
            path_artist, path_song = parts[-2], os.path.splitext(parts[-1])[0]
            if self.fuzzy_match(artist, path_artist) and self.fuzzy_match(title, path_song):
                matches.append(path)
        return matches    

    @staticmethod
    def fuzzy_match(s1: str, s2: str, threshold: int = 80) -> bool:
        return fuzz.ratio(s1.lower(), s2.lower()) > threshold

if __name__ == '__main__':
    lakh_clean_searcher = LakhCleanSearcher()
    lakh_clean_searcher.search_directory()

