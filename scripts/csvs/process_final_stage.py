import pandas as pd
import os

from typing import Optional, Dict, Any

from src.defaults import *

"""
From the validated CSV, we want to 
- ensure a song doesn't get multiple IDs due to being part of multiple court cases
"""