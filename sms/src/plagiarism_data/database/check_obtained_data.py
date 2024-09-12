import sqlite3
import sys

from sms.defaults import *

def check_obtained_data(print_cases: bool = False) -> None:
    """Checks the obtained data in the database."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    query = """
    SELECT 
        cc.case_id,
        s1.artist as complaining_artist,
        s1.title as complaining_title,
        s2.artist as defending_artist,
        s2.title as defending_title,
        cc.case_won,
        cc.is_melodic_comparison
    FROM 
        copyright_cases cc
    JOIN 
        copyright_songs s1 ON cc.complaining_id = s1.song_id
    JOIN 
        copyright_songs s2 ON cc.defending_id = s2.song_id
    WHERE 
        EXISTS(SELECT 1 FROM song_paths WHERE song_id = cc.complaining_id)
        AND EXISTS(SELECT 1 FROM song_paths WHERE song_id = cc.defending_id)
    """

    cursor.execute(query)
    cases = cursor.fetchall()

    total_valid_pairs = len(cases)
    won_cases = sum(1 for case in cases if case[5])  # case[5] is case_won
    won_melodic_cases = sum(1 for case in cases if case[5] and case[6])  # case[6] is is_melodic_comparison

    print(f"Total valid pairs: {total_valid_pairs}")
    print(f"Won cases: {won_cases}")
    print(f"Won melodic cases: {won_melodic_cases}")

    if print_cases:
        print("\nCase details:")
        for case in cases:
            case_id, comp_artist, comp_title, def_artist, def_title, case_won, is_melodic = case
            print(f"Case ID: {case_id}")
            print(f"Complaining: {comp_artist} - {comp_title}")
            print(f"Defending: {def_artist} - {def_title}")
            print(f"Case won: {'Yes' if case_won else 'No'}")
            print(f"Is melodic comparison: {'Yes' if is_melodic else 'No'}")
            print("-" * 40)

    conn.close()
        
if __name__ == '__main__':
    print_cases = 'print' in sys.argv
    check_obtained_data(print_cases)