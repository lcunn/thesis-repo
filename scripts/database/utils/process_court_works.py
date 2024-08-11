from pydantic import BaseModel
from openai import OpenAI
import pandas as pd

from src.defaults import *
from scripts.database.produce_copyright_songs_base_csv import _reformat_songs_df

client = OpenAI()

class MBOutput(BaseModel):
    artist: str
    title: str
    confidence: float

SYSTEM_PROMPT = """
You are an expert at music copyright history. 
We have a list of songs that have appeared in music copyright cases. The end goal is, starting from the court description of the title and artist of a song, get the commonly known artist and title of a song.
This is because we then want to search for this song using the MusicBrainz API.
You will receive four fields of information:
- case_year: the year of the copyright case
- case_name: the name of the copyright case
- complaining/defending_work: the court-given title of the complaining/defending work
- complaining_author/defending: the court-given name of the complaining_author/defenders.
Your job is, using analytical thinking and your own knowledge of copyright history, to extract the title and artist strings that will be likely to come up in the MusicBrainz database. 
You will also give a confidence, between 0 and 1, of how likely you think it is that you're right.

Example:

Prompt:
case_year: 2019
case_name: Batiste v. Lewis, et al.
complaining/defending_work: Thrift Shop; Neon Cathedral
complaining_author/defending: Ryan Lewis; Ben Haggerty ("Macklemore")

Output:
artist: Macklemore & Ryan Lewis feat. Wanz
title: Thrift Shop
confidence: 0.99
"""

def _format_user_prompt(year: str, name: str, work: str, author: str) -> str:
    prompt = f"""
    case_year: {year}
    case_name: {name}
    complaining/defending_work: {work}
    complaining_author/defending: {author}
    """
    return prompt

def process_single_case(year: str, name: str, work: str, author: str):

    user_prompt = _format_user_prompt(year, name, work, author)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=MBOutput,
    )

    event = completion.choices[0].message.parsed
    return event

def process_cases(df: pd.DataFrame, filter: bool = True):
    """
    Expects a DataFrame with the following columns:
        case_id
        year
        case_name
        case_artist
        case_title

    Returns the same DataFrame with the following columns added:
        gpt_artist
        gpt_title
        gpt_conf
    """
    gpt_df = df.copy(deep=True)
    df['gpt_artist'] = None
    df['gpt_title'] = None
    df['gpt_conf'] = None

    for i, r in gpt_df.iterrows():
        user_prompt = _format_user_prompt(r['year'], r['case_name'], r['case_title'], r['case_artist'])
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=MBOutput,
        )
        event = completion.choices[0].message.parsed
        gpt_df.loc[i, ['gpt_artist', 'gpt_title', 'gpt_conf']] = event.artist, event.title, event.confidence

    if filter:
        gpt_df = gpt_df[~(gpt_df['gpt_artist'].isna() | gpt_df['gpt_title'].isna())]

    return gpt_df
