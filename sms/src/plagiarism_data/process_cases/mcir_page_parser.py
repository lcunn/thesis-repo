import requests
from bs4 import BeautifulSoup

# page parser to scrape comments from https://blogs.law.gwu.edu/mcir/case/...

def fetch_webpage_content(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.content, 'html.parser')

def extract_comments_and_opinions(soup: BeautifulSoup) -> list:
    h3_elements = soup.find_all('h3')
    comments_and_opinions = []

    for h3 in h3_elements:
        text = h3.text.lower()
        if 'comment' in text or 'opinion' in text:
            content = []
            for sibling in h3.next_siblings:
                if sibling.name == 'h3':
                    break
                if sibling.name == 'p':
                    content.append(sibling.text.strip())
            comments_and_opinions.append(' '.join(content))

    return comments_and_opinions

def mcir_page_parser(url: str) -> list:
    soup = fetch_webpage_content(url)
    return extract_comments_and_opinions(soup)
