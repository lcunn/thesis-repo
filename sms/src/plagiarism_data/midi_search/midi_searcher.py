import os
import datetime
import shutil
import logging
import sqlite3
from typing import List
from abc import ABC, abstractmethod

from sms.defaults import *
from sms.src.log import configure_logging

class MidiSearcher(ABC):

    def __init__(self, db_path: str = DATABASE):
    
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.set_search_path()
        log_path = self.get_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        configure_logging(to_console=True, to_file=True, log_file=log_path)
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        self.conn.close()

    @abstractmethod
    def set_search_path(self) -> None:
        """Set self.source_name and self.directory."""
        pass

    @abstractmethod
    def find_song_matches(self, artist: str, title: str) -> List[str]:
        """
        Finds the matches for the given artist and title in the search directory.
        Should return a list of full file paths.      
        """
        pass

    def get_log_path(self) -> str:

        today = datetime.date.today().strftime("%Y-%m-%d")
        base_path = os.path.join(MIDI_SEARCH_LOG_PATH, self.source_name)
        os.makedirs(base_path, exist_ok=True)
        # Check for existing log files for today
        existing_logs = [f for f in os.listdir(base_path) if f.startswith(today)]
        count = len(existing_logs) + 1
        
        log_filename = f"{today}_{count}.log"
        return os.path.join(base_path, log_filename)

    def register_match_in_copyright_folder(
            self, 
            source_name: str, 
            match: str, 
            song_id: int, 
            song_title: str, 
            song_artist: str,
            suffix_num: int = 1
        ) -> None:

        # make destination filename and path
        dest_filename = f"{source_name}_{song_artist}_{song_title}_{suffix_num}.mid".replace(" ", "_")
        dest_filename = ''.join(c for c in dest_filename if c.isalnum() or c in ['_', '.'])
        dest_path = os.path.join(COPYRIGHT_MIDI_PATH, str(song_id), dest_filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # check if file already exists in the destination
        if os.path.exists(dest_path):
            self.logger.info(f"File already exists at {dest_path}. Skipping copy.")
            return

        # try to copy file to destination
        try:
            shutil.copy2(match, dest_path)
            self.logger.info(f"Copied {match} to {dest_path}")
            self.cursor.execute('''
                INSERT OR REPLACE INTO song_paths (song_id, file_path, source)
                VALUES (?, ?, ?)
            ''', (song_id, dest_path, source_name))
            self.conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to copy {match} to {dest_path}: {str(e)}")
        
    def search_directory(self) -> None:

        # get copyright songs
        self.cursor.execute("SELECT song_id, artist, title FROM copyright_songs")

        # for each song in the database
        for song_id, artist, title in self.cursor.fetchall():
            self.logger.debug(f'Searching for {artist} - {title}')

            # find matches in the directory using subclass-defined method
            matches: List[str] = self.find_song_matches(artist, title)
            for i, match in enumerate(matches):
                # format match for logging
                match_parts = match.split(os.sep)
                shortened_match = os.path.join(*match_parts[-3:]) if len(match_parts) > 3 else match
                # log the match
                self.logger.info(f"Song ID: {song_id}, Artist: {artist}, Title: {title}, Match: {shortened_match}")
                self.register_match_in_copyright_folder(self.source_name, match, song_id, title, artist, i+1)

        self.logger.info(f"Finished searching {self.source_name}")