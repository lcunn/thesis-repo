from pydantic import BaseModel
from openai import OpenAI
import pandas as pd

from src.defaults import *

client = OpenAI()

class Evaluation(BaseModel):
    correctness_category: int

SYSTEM_PROMPT = """
We have a table about music copyright cases. We started with a list of copyright cases, then you processed it to extract the title and artist of the song, and then we search MusicBrainz for their ID of it.
You need is to validate the MusicBrainz search, which has been done by using the API with the gpt_title and gpt_artist. Here are the fields of the song you will receive:
- case_year: the year of the copyright case
- case_name: the name of the copyright case
- complaining/defending_work: the court-given title of the complaining/defending work
- complaining_author/defending: the court-given name of the complaining_author/defenders.
- gpt_title: the song title extracted from the complaining/defending_work
- gpt_artist: the artist extracted from the complaining_author/defending
- gpt_conf: the confidence with which the extraction is judged to match the song referenced in the case
- mb_title: the title obtained from searching the MusicBrainz catalogue
- mb_artist: the artist obtained from searching the MusicBrainz catalogue

Your job will be to categorise the mb_title and mb_artist by how well it matches the case, i.e. the title, artist in complaining/defending_work, complaining_author/defending.
While doing this, be aware of the pipeline that happened. We went from the case, to the gpt_ fields, which are the most likely songs from the case, then to the mb_, which is from the MB database.
You need to return the category (an integer) that the song best belongs to:
1: the mb_title and mb_artist very clearly (i.e. almost identically) match the case; very high confidence
2: the mb_title and mb_artist are similar to the case (i.e. they can quite easily be extracted from the case); high confidence
3: there are small inaccuracies in each step; medium confidence
4: the gpt_title and gpt_title seem well extracted from the case, but it hasn't been found in the mb database.
5: the case hasn't been well extracted into the gpt_, and thus hasnt been found by mb.
Use your analytical capabilities and your copyright knowledge.

Example:

Prompt:
case_year: 2019
case_name: Batiste v. Lewis, et al.
complaining/defending_work: Thrift Shop; Neon Cathedral
complaining_author/defending: Ryan Lewis; Ben Haggerty ("Macklemore")
gpt_artist: Macklemore & Ryan Lewis feat. Wanz
gpt_title: Thrift Shop
gpt_conf: 0.99
mb_title: Thrift Shop
mb_artist: Macklemore & Ryan Lewis feat. Wanz

Output:
correctness_category: 2
"""

def _format_user_prompt(year: str, name: str, work: str, author: str, gpt_artist: str, gpt_title: str, gpt_conf: str, mb_title: str, mb_artist: str) -> str:
    prompt = f"""
    case_year: {year}
    case_name: {name}
    complaining/defending_work: {work}
    complaining_author/defending: {author}
    gpt_artist: {gpt_artist}
    gpt_title: {gpt_title}
    gpt_conf: {gpt_conf}
    mb_title: {mb_title}
    mb_artist: {mb_artist}
    """
    return prompt

def validate_mb_results(df: pd.DataFrame):
    """
    Expects a DataFrame with the following columns:
        case_id
        year
        case_name
        case_artist
        case_title
        gpt_artist
        gpt_title
        gpt_conf
        mb_title
        mb_artist

    Returns the same DataFrame with the following columns added:
        correctness_category
        
    """
    mb_df = df.copy(deep=True)
    mb_df['correctness_category'] = None

    for i, r in mb_df.iterrows():
        user_prompt = _format_user_prompt(
            r['year'], 
            r['case_name'], 
            r['case_title'], 
            r['case_artist'], 
            r['gpt_artist'], 
            r['gpt_title'], 
            r['gpt_conf'], 
            r['mb_title'], 
            r['mb_artist']
            )
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=Evaluation,
        )
        event = completion.choices[0].message.parsed
        mb_df.loc[i, ['correctness_category']] = event.correctness_category

    return mb_df