import requests
from bs4 import BeautifulSoup
import pandas as pd

from sms.defaults import *

def produce_table(path: str = COPYRIGHT_CLAIMS_CSV):
    # URL of the page
    url = "https://blogs.law.gwu.edu/mcir/cases-2/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Locate the table
    table = soup.find('table', {'id': 'tablepress-4'})

    # Extract headers from the first row of the tbody
    headers = [th.text.strip() for th in table.find('tbody').find('tr').find_all('td')]

    # Extract rows, skipping the first row since it's the header row
    rows = []
    for row in table.find_all('tr')[2:]:  # Start from the second row in the tbody
        cols = row.find_all('td')
        row_data = [col.text.strip() for col in cols]

        # Get the case name and link
        case_name_tag = row.find('a')
        if case_name_tag:
            case_name = case_name_tag.text.strip()
            case_link = case_name_tag['href']
            row_data[2] = case_name  # Update the case name
            row_data.insert(3, case_link)  # Insert link after case name
        else:
            case_link = None
            row_data.insert(3, case_link)  # No link available

        rows.append(row_data)

    headers = headers[:2] + ['Case Name', 'Link'] + headers[3:]
    headers = [header.lower().replace(' ', '_') for header in headers]
    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(path, index_label='case_id')

if __name__ == "__main__":
    produce_table()