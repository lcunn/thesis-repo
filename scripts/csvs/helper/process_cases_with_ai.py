from pydantic import BaseModel
from openai import OpenAI
import pandas as pd

from src.defaults import *
from scripts.csvs.helper.ai_prompts import *

client = OpenAI()

class ai_output(BaseModel):
    artist: str
    title: str
    confidence: float

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
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ai_output,
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
