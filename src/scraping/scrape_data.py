# Reverse engineering VLR player rating
# Purpose: To reverse engineer the VLR.gg player rating calculation using their stat data.
# File: scrape_data.py - used to parse vlr stats pages and store raw data.
# Warren Xu

import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

base_url = "https://www.vlr.gg"

# VLR's /stats filters are span (30d/60d/90d/<year>/all), tier, region and min_rounds.
# Unknown params are silently ignored, so a typo here returns the default table.
pages = {
    "champions2023": "/event/stats/1657/valorant-champions-2023",
    "season2023": "/stats/?tier=vct&region=all&span=2023&min_rounds=400&min_rating=1550",
    "stats_alltime": "/stats/?tier=vct&region=all&span=all&min_rounds=1000&min_rating=1550",
    "stats_recent": "/stats/?tier=vct&region=all&span=90d&min_rounds=200&min_rating=1550",
}

# Each <td> carries a data-col attribute; map those to our column names.
# FK% / FD% on VLR are first kills / deaths per round, so they become FKPR / FDPR.
columns = {
    "rnd": "Rnd",
    "rating2": "R2.0",
    "acs": "ACS",
    "kd": "K:D",
    "kast": "KAST",
    "adr": "ADR",
    "kpr": "KPR",
    "apr": "APR",
    "fbpr": "FKPR",
    "fdpr": "FDPR",
    "hsp": "HS%",
    "clp": "CL%",
    "cl": "CL",
    "kmax": "KMax",
    "k": "K",
    "d": "D",
    "a": "A",
    "fk": "FK",
    "fd": "FD",
}

REQUEST_DELAY = 1.0  # seconds between requests, to be polite to vlr.gg


def make_session():
    # /stats sets an "abok" cookie and 302s back to bare /stats (dropping the query string)
    # on the first visit, so prime the cookie before requesting any filtered page.
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (vlr-ranking-formula research scraper)"
    session.get(base_url + "/stats").raise_for_status()
    return session


def parse_table(soup):
    table = soup.find("table", id="st-table")
    if table is None:
        raise ValueError("stats table not found - vlr.gg markup may have changed")

    rows = []
    for tr in table.find_all("tr")[1:]:
        team = tr.find(class_="st-pl-country")  # missing for free agents
        row = {
            "Player": tr.find(class_="st-pl-name").get_text(strip=True),
            "Team": team.get_text(strip=True) if team else None,
        }
        for td in tr.find_all("td", attrs={"data-col": True}):
            if td["data-col"] in columns:
                row[columns[td["data-col"]]] = td.get_text("", strip=True)
        rows.append(row)
    return rows


def scrape_page(session, path):
    # /stats is paginated 100 players at a time; follow the pager until there's no next page.
    rows = []
    page = 1
    while True:
        response = session.get(base_url + path, params={"page": page} if page > 1 else None)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        rows.extend(parse_table(soup))

        if soup.find("a", href=lambda href: href and f"page={page + 1}" in href) is None:
            return rows
        page += 1
        time.sleep(REQUEST_DELAY)


session = make_session()

for page_name, path in pages.items():
    try:
        rows = scrape_page(session, path)
        df = pd.DataFrame(rows, columns=["Player", "Team", *columns.values()])
        csv_file = f"data/raw/{page_name}.csv"
        df.to_csv(csv_file, index=False)

        print(f"Saved {len(df)} players into {csv_file}")

    except (requests.RequestException, ValueError) as e:
        print(f"Failed to retrieve data for {page_name}: {e}")

    time.sleep(REQUEST_DELAY)

print("\nScraping data for all pages completed.")
