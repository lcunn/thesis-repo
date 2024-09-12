import sqlite3
import pandas as pd
from sms.defaults import *

def create_connection():
    conn = sqlite3.connect(DATABASE)
    return conn

def create_tables(conn):
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS copyright_songs (
        song_id INTEGER PRIMARY KEY,
        artist TEXT,
        title TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS copyright_cases (
        case_id INTEGER PRIMARY KEY,
        complaining_id INTEGER,
        defending_id INTEGER,
        is_melodic_comparison BOOLEAN,
        case_won BOOLEAN,
        FOREIGN KEY (complaining_id) REFERENCES copyright_songs (song_id),
        FOREIGN KEY (defending_id) REFERENCES copyright_songs (song_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS song_paths (
        song_id INTEGER,
        file_path TEXT,
        source TEXT,
        PRIMARY KEY (song_id, file_path),
        FOREIGN KEY (song_id) REFERENCES copyright_songs (song_id)
    )
    ''')
    
    conn.commit()

def table_is_empty(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    return count == 0

def insert_songs(conn, df):
    if table_is_empty(conn, 'copyright_songs'):
        df.to_sql('copyright_songs', conn, if_exists='replace', index=False)
        print("Inserted data into copyright_songs table.")
    else:
        print("copyright_songs table already contains data. Skipping insertion.")

def insert_cases(conn, df):
    if table_is_empty(conn, 'copyright_cases'):
        df.to_sql('copyright_cases', conn, if_exists='replace', index=False)
        print("Inserted data into copyright_cases table.")
    else:
        print("copyright_cases table already contains data. Skipping insertion.")

def main():
    conn = create_connection()
    create_tables(conn)
    
    songs_df = pd.read_csv(COPYRIGHT_SONGS_CSV)
    cases_df = pd.read_csv(COPYRIGHT_PAIRS_CSV)
    
    insert_songs(conn, songs_df)
    insert_cases(conn, cases_df)
    
    conn.close()
    print("Database created and populated successfully.")

if __name__ == "__main__":
    main()