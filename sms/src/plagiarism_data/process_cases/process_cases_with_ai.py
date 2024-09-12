from typing import List
import logging
import os

from pydantic import BaseModel
from openai import OpenAI
import pandas as pd
import requests
from bs4 import BeautifulSoup

from sms.defaults import *
from sms.src.plagiarism_data.process_cases.ai_prompts import GPT_SYSTEM_PROMPT
from sms.src.plagiarism_data.process_cases.mcir_page_parser import mcir_page_parser

logger = logging.getLogger(__name__)

client = OpenAI()

class Song(BaseModel):
    artist: str
    title: str
    evidence: str

class Pair(BaseModel):
    song1: Song
    song2: Song
    pair_evidence: str
    is_melodic_comparison: bool
    melodic_evidence: str
    was_case_won: bool
    case_won_evidence: str
    
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
    """.encode('utf-8').decode('unicode_escape')
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

def estimate_songs_from_cases(file_path: str = COPYRIGHT_CLAIMS_CSV_F):

    df = pd.read_csv(file_path)
    
    # Check if checkpoint file exists and load it
    if os.path.exists(GPT_SONGS_CHECKPOINT_CSV):
        pairs_df = pd.read_csv(GPT_SONGS_CHECKPOINT_CSV)
        processed_case_ids = set(pairs_df['case_id'])
    else:
        pairs_df = pd.DataFrame(columns=['case_id', 
                                         'song1_artist', 
                                         'song1_title', 
                                         'song1_evidence',
                                         'song2_artist', 
                                         'song2_title', 
                                         'song2_evidence', 
                                         'pair_evidence',
                                         'is_melodic_comparison', 
                                         'melodic_evidence', 
                                         'was_case_won', 
                                         'case_won_evidence'])
        processed_case_ids = set()

    for _, row in df.iterrows():
        if row['case_id'] in processed_case_ids:
            continue  # Skip already processed cases

        logger.info(f"Processing case {row['case_id']}")
        try:
            output = process_single_case(row)
            for pair in output.pairs:
                new_row = pd.DataFrame({
                    'case_id': [row['case_id']],
                    'song1_artist': [pair.song1.artist],
                    'song1_title': [pair.song1.title],
                    'song1_evidence': [pair.song1.evidence],
                    'song2_artist': [pair.song2.artist],
                    'song2_title': [pair.song2.title],
                    'song2_evidence': [pair.song2.evidence],
                    'pair_evidence': [pair.pair_evidence],
                    'is_melodic_comparison': [pair.is_melodic_comparison],
                    'melodic_evidence': [pair.melodic_evidence],
                    'was_case_won': [pair.was_case_won],
                    'case_won_evidence': [pair.case_won_evidence]
                })
                pairs_df = pd.concat([pairs_df, new_row], ignore_index=True)
            
            # Save the DataFrame after each successful iteration
            pairs_df.to_csv(GPT_SONGS_CHECKPOINT_CSV, index=False)
            processed_case_ids.add(row['case_id'])
        except Exception as e:
            logger.error(f"Error processing case {row['case_id']}: {str(e)}")

    return pairs_df