from pydantic import BaseModel
from typing import List
import logging
from openai import OpenAI
import pandas as pd
import requests
from bs4 import BeautifulSoup

from plagdet.src.defaults import *
from plagdet.scripts.csvs.process_cases.ai_prompts import GPT_SYSTEM_PROMPT
from plagdet.scripts.csvs.process_cases.mcir_page_parser import mcir_page_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI()

class Song(BaseModel):
    artist: str
    title: str
    confidence: float
    evidence: List[str]
    evidence_source: str

class Pair(BaseModel):
    song1: Song
    song2: Song
    is_melodic_comparison: bool
    evidence: List[str]
    evidence_source: str
    
class AIOutput(BaseModel):
    pairs: List[Pair]

def format_user_prompt(year: str, name: str, court: str, complaining_work: str, defending_work: str, complaining_author: str, defending: str, link: str) -> str:
    comments_and_opinions = mcir_page_parser(link)
    prompt = f"""
    year: {year}
    case_name: {name}
    court: {court}
    complaining_work: {complaining_work}
    defending_work: {defending_work}
    complaining_author: {complaining_author}
    defending: {defending}
    comments_and_opinions: {comments_and_opinions}
    """
    return prompt

def process_single_case(row: pd.Series):
    user_prompt = format_user_prompt(
        row['year'], row['case_name'], row['court'],
        row['complaining_work'], row['defending_work'],
        row['complaining_author'], row['defending'],
        row['link']
    )
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=AIOutput
    )
    output = AIOutput.parse_raw(completion.choices[0].message.content)
    return output

def process_cases(df: pd.DataFrame):
    pairs_df = pd.DataFrame(columns=['case_id', 'song1_artist', 'song1_title', 'song1_confidence', 'song1_evidence',
                                     'song2_artist', 'song2_title', 'song2_confidence', 'song2_evidence',
                                     'is_melodic_comparison', 'pair_evidence', 'pair_evidence_source'])

    for _, row in df.iterrows():
        logger.info(f"Processing case {row['case_id']}")
        output = process_single_case(row)
        for pair in output.pairs:
            pairs_df = pairs_df.append({
                'case_id': row['case_id'],
                'song1_artist': pair.song1.artist,
                'song1_title': pair.song1.title,
                'song1_confidence': pair.song1.confidence,
                'song1_evidence': pair.song1.evidence,
                'song1_evidence_source': pair.song1.evidence_source,
                'song2_artist': pair.song2.artist,
                'song2_title': pair.song2.title,
                'song2_confidence': pair.song2.confidence,
                'song2_evidence': pair.song2.evidence,
                'song2_evidence_source': pair.song2.evidence_source,
                'is_melodic_comparison': pair.is_melodic_comparison,
                'pair_evidence': pair.evidence,
                'pair_evidence_source': pair.evidence_source
            }, ignore_index=True)

    return pairs_df
