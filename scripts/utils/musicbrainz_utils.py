import pandas as pd
import sqlite3
import musicbrainzngs
import os
import time

from typing import Optional, Dict, Any
from src.defaults import *
from src.private_defaults import MUSICBRAINZ_AGENT

def set_musicbrainz_api() -> None:
    """Set musicbrainz api agent."""
    musicbrainzngs.set_useragent(*list(MUSICBRAINZ_AGENT.values()))

def search_musicbrainz(artist: str, title: str) -> Optional[Dict[str, Any]]:
    """Search musicbrainz for a song by artist and title."""
    result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
    if result['recording-list']:
        return result['recording-list'][0]
    return None

