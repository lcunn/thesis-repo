from pydantic import BaseModel
from openai import OpenAI
import pandas as pd

from src.defaults import *

client = OpenAI()

class MBOutput(BaseModel):
    artist: str
    title: str
    confidence: float

SYSTEM_PROMPT = """
You are an expert at music copyright history. 
We have a list of songs that have appeared in music copyright cases. The end goal is, starting from the court description of the titles and artists of the songs involved, to get the commonly known artists and titles of these songs.
This is because we then want to search for these songs using the MusicBrainz API.
You will receive the following fields of information:
- year: the year of the copyright case
- country: the country where the case was filed
- case_name: the name of the copyright case
- link: a link to more information about the case
- court: the court where the case was filed
- complaining_work: the court-given title of the complaining work
- defending_work: the court-given title of the defending work
- complaining_author: the court-given name of the complaining author
- defending: the court-given name of the defenders
- case_extract: a brief extract or comment about the case from the provided link

Your job is, using analytical thinking and your own knowledge of copyright history, to extract the title and artist strings for both the complaining and defending works that will be likely to come up in the MusicBrainz database. 
You will also give a confidence, between 0 and 1, of how likely you think it is that you're right for each work.

Example:

Input:
year: 2019
country: US
case_name: Batiste v. Lewis, et al.
link: https://blogs.law.gwu.edu/mcir/case/inplay-batiste-v-ben-haggerty-aka-macklemore-et-al/
court: E.D. La.
complaining_work: Hip Jazz; World of Blues; Salsa 4 Elise; I Got the Rhythm On
defending_work: Thrift Shop; Neon Cathedral
complaining_author: Paul Batiste
defending: Ryan Lewis; Ben Haggerty ("Macklemore")
case_extract: The case involves allegations of copyright infringement against Macklemore & Ryan Lewis for their hit songs "Thrift Shop" and "Neon Cathedral".

Output:
complaining_work:
  artist: Paul Batiste
  title: Hip Jazz
  confidence: 0.7
defending_work:
  artist: Macklemore & Ryan Lewis
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

def process_cases(df: pd.DataFrame):
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

    return gpt_df
