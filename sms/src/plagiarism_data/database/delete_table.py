import sqlite3
from sms.defaults import *

def delete_table(table_name: str) -> None:
    """Deletes the specified table from DATABASE."""
    # Connect to SQLite database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Execute the DROP TABLE command
    cursor.execute(f'DROP TABLE IF EXISTS {table_name}')

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print(f"Table '{table_name}' has been deleted from the database.")

if __name__ == "__main__":
    import sys
    delete_table(str(sys.argv[1]))
