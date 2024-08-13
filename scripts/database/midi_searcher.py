import os
import shutil
import logging
import pandas as pd
from typing import Dict, Union, Any, Protocol, List
from src.defaults import *

class ComparisonFunction(Protocol):
    def __call__(self, dir_list: List[str], artist: str, title: str) -> List[str]:
        ...

"""
SEARCH_CONFIGS is a dictionary of search functions. 
The key is the name of child directory within MIDI_DATA_PATH.
The value is a function which compares a song title and artist to the formatted directory list and returns matches.
A match is the path to the midi file from the directory.
"""

class MidiSearcher:

    def __init__(self, search_configs: Dict[str, Dict[str, Any]], csv_path: str = COPYRIGHT_SONGS_ALL_FIELDS):
        self.df = pd.read_csv(csv_path)
        self.search_configs = search_configs

    def _update_csv(self):
        self.df.to_csv(COPYRIGHT_SONGS_ALL_FIELDS, index=False)

    def _make_directory_list(self, directory: str) -> List[str]:
        filepaths = []
        for root, directories, files in os.walk(LAKH_CLEAN_PATH):
            for file in files:
                filepaths.append(os.path.join(root, file))
        return filepaths

    def _find_song_matches(self, directory: str, directory_list: List[str], artist: str, title: str) -> List[str]:
        """
        Takes the directory, processes it into a list of comparable strings, and compares it to the song title and artist.

        Returns:
            matches: list of midi paths that match the song title and artist.
        """
        c_func: ComparisonFunction = self.search_configs[directory]
        matches = c_func(directory_list, artist, title)
        return matches
    
    def _register_match_in_copyright_folder(self, directory: str, match: str, song_id: str, song_title: str, song_artist: str):

        # Create the destination filename
        dest_filename = f"{directory}_{song_artist}_{song_title}.mid".replace(" ", "_")
        dest_filename = ''.join(c for c in dest_filename if c.isalnum() or c in ['_', '.'])  # Remove any invalid characters
        # Create the full destination path
        dest_path = os.path.join(COPYRIGHT_MIDI_PATH, str(song_id), dest_filename)
        # Ensure the song_id subdirectory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # Copy the file
        try:
            shutil.copy2(match, dest_path)
            logging.info(f"Copied {match} to {dest_path}")
        except Exception as e:
            logging.error(f"Failed to copy {match} to {dest_path}: {str(e)}")

    def search_directory(self, directory: str):
        """
        Searches a directory for midi matches, adds them to the CSV file, and adds them to the copyright folder.
        """
        # check for column/instantiate column
        if directory in self.df.columns:
            logging.info(f'Directory {directory} has been processed before.')
            proceed = input('Do you want to proceed? (y/n)')
            if proceed != 'y':
                return
        else:
            self.df[directory] = [[] for _ in range(len(self.df))]

        directory_list = self._make_directory_list(directory)
        for i, row in self.df.iterrows():
            logging.info(f'Searching for {row["gpt_artist"]} - {row["gpt_title"]}')
            # find matches
            matches = self._find_song_matches(directory, directory_list, row['gpt_artist'], row['gpt_title'])
            # if not already, register match and deposit in copyright folder
            for match in matches:
                logging.info(f'Match found: {match}')
                if match not in row[directory]:
                    row[directory].append(match)
                    self._register_match_in_copyright_folder(directory, match, row['song_id'], row['gpt_title'], row['gpt_artist'])
        
        self._update_csv()










