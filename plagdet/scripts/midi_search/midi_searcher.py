import os
import shutil
import logging
import sqlite3
from typing import List
from abc import ABC, abstractmethod
from src.defaults import *

class MidiSearcher(ABC):
    def __init__(self, db_path: str = DATABASE):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    @abstractmethod
    def find_song_matches(self, directory_list: List[str], artist: str, title: str) -> List[str]:
        pass

    def make_directory_list(self, directory: str) -> List[str]:
        filepaths = []
        for root, _, files in os.walk(directory):
            for file in files:
                filepaths.append(os.path.join(root, file))
        return filepaths

    def register_match_in_copyright_folder(
            self, 
            source_name: str, 
            match: str, 
            song_id: int, 
            song_title: str, 
            song_artist: str
        ) -> None:

        # make destination filename and path
        dest_filename = f"{source_name}_{song_artist}_{song_title}.mid".replace(" ", "_")
        dest_filename = ''.join(c for c in dest_filename if c.isalnum() or c in ['_', '.'])
        dest_path = os.path.join(COPYRIGHT_MIDI_PATH, str(song_id), dest_filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # try to copy file to destination
        try:
            shutil.copy2(match, dest_path)
            logging.info(f"Copied {match} to {dest_path}")
            self.cursor.execute('''
                INSERT OR REPLACE INTO song_paths (song_id, file_path, source)
                VALUES (?, ?, ?)
            ''', (song_id, dest_path, source_name))
            self.conn.commit()
        except Exception as e:
            logging.error(f"Failed to copy {match} to {dest_path}: {str(e)}")

    def search_directory(self, directory: str, source_name: str):
        directory_list = self.make_directory_list(directory)
        self.cursor.execute("SELECT song_id, artist, title FROM copyright_songs")
        for song_id, artist, title in self.cursor.fetchall():
            logging.info(f'Searching for {artist} - {title}')
            matches = self.find_song_matches(directory_list, artist, title)
            for match in matches:
                self.register_match_in_copyright_folder(source_name, match, song_id, title, artist)
        logging.info(f"Finished searching {source_name}")

class LakhCleanSearcher(MidiSearcher):
    def find_song_matches(self, directory_list: List[str], artist: str, title: str) -> List[str]:
        matches = []
        for path in directory_list:
            parts = os.path.normpath(path).split(os.sep)
            if len(parts) < 2:
                continue
            path_artist, path_song = parts[-2], os.path.splitext(parts[-1])[0]
            if self.fuzzy_match(artist, path_artist) and self.fuzzy_match(title, path_song):
                matches.append(path)
        return matches

    @staticmethod
    def fuzzy_match(s1: str, s2: str, threshold: int = 80) -> bool:
        from fuzzywuzzy import fuzz
        return fuzz.ratio(s1.lower(), s2.lower()) > threshold










