import pandas as pd
from fuzzywuzzy import fuzz
from sms.defaults import *

def is_empty(value):
    if isinstance(value, pd.Series):
        return value.isna() | (value.astype(str).str.strip() == '')
    return pd.isna(value) or str(value).strip() == ''

def are_works_similar(work1, work2, threshold=95):
    if is_empty(work1) or is_empty(work2):
        return False
    return fuzz.ratio(str(work1), str(work2)) >= threshold

def filter_unwanted_cases(csv_path: str = COPYRIGHT_CLAIMS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Filter cases with only one song or similar works
    filtered_df = df[
        (~is_empty(df['complaining_work']) & ~is_empty(df['defending_work'])) |
        (is_empty(df['complaining_work']) ^ is_empty(df['defending_work']))
    ]
    
    # Remove cases with similar complaining and defending works
    filtered_df = filtered_df[
        ~(~is_empty(filtered_df['complaining_work']) & 
          ~is_empty(filtered_df['defending_work']) & 
          filtered_df.apply(lambda row: are_works_similar(row['complaining_work'], row['defending_work']), axis=1))
    ]
    return filtered_df

# def reformat_copyright_claims(cases_df: pd.DataFrame) -> pd.DataFrame:
    
#     df = cases_df.copy(deep=True)
#     complainers = df[['case_id', 'year', 'case_name', 'complaining_author', 'complaining_work']].rename(
#         columns={'complaining_author': 'case_artist', 'complaining_work': 'case_title'})
#     complainers['case_role'] = 'complainer'
    
#     defendants = df[['case_id', 'year', 'case_name', 'defending', 'defending_work']].rename(
#         columns={'defending': 'case_artist', 'defending_work': 'case_title'})
#     defendants['case_role'] = 'defendant'

#     songs = pd.concat([complainers, defendants], ignore_index=True)
#     return songs

# def process_into_songs(csv_path: str = COPYRIGHT_CLAIMS_CSV):

#     wanted_cases = filter_unwanted_cases(csv_path)
#     songs = reformat_copyright_claims(wanted_cases)
#     return songs