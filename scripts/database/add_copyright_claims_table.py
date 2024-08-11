import pandas as pd
import sqlite3
import argparse
import sys

from src.defaults import *

TABLE_NAME = 'copyright_claims'

def produce_copyright_claims_table() -> None:
    """Write the copyright_claims csv to the DATABASE"""
    df = pd.read_csv(COPYRIGHT_CLAIMS_CSV)
    conn = sqlite3.connect(DATABASE)
    df.to_sql(TABLE_NAME, conn, if_exists='replace', index=False, 
              dtype={"id": "INTEGER PRIMARY KEY"})
    conn.commit()
    conn.close()
    print(f"Data imported successfully into table '{TABLE_NAME}'")

if __name__ == '__main__':
    produce_copyright_claims_table()