import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Target URL
URL = 'https://s2025.conference-schedule.org'

# Required columns mapping
REQUIRED_COLUMNS = ['title', 'date', 'time', 'type', 'contributors', 'location']


def fetch_page(url):
    """Fetch the webpage content with error handling."""
    try:
        print(f"Fetching URL: {url}")
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        print("Page fetched successfully.")
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        raise SystemExit(1)


def parse_schedule(html):
    """Parse the schedule and extract required columns into a list of dicts."""
    soup = BeautifulSoup(html, 'lxml')
    schedule_data = []
    current_date = None

    # Find all date headings (e.g., 'Sunday, 10 August 2025')
    for heading in soup.find_all(['h2', 'h3', 'div', 'span']):
        if heading.get_text(strip=True).endswith('2025'):
            current_date = heading.get_text(strip=True)
        # Find the table or grid immediately following the date heading
        next_sibling = heading.find_next_sibling()
        if next_sibling:
            # Try to find a table
            table = next_sibling if next_sibling.name == 'table' else next_sibling.find('table')
            if table:
                # Use pandas to read the table
                try:
                    dfs = pd.read_html(str(table))
                except Exception as e:
                    print(f"Error parsing table with pandas: {e}")
                    continue
                for df in dfs:
                    # Map columns
                    col_map = {}
                    for col in df.columns:
                        col_lower = str(col).strip().lower()
                        if 'session' in col_lower or 'presentation' in col_lower:
                            col_map['title'] = col
                        elif 'time' in col_lower:
                            col_map['time'] = col
                        elif 'type' in col_lower:
                            col_map['type'] = col
                        elif 'contributor' in col_lower:
                            col_map['contributors'] = col
                        elif 'location' in col_lower:
                            col_map['location'] = col
                    # Check if all required columns are present
                    if all(k in col_map for k in ['title', 'time', 'type', 'contributors', 'location']):
                        for _, row in df.iterrows():
                            entry = {
                                'title': row[col_map['title']],
                                'date': current_date,
                                'time': row[col_map['time']],
                                'type': row[col_map['type']],
                                'contributors': row[col_map['contributors']],
                                'location': row[col_map['location']]
                            }
                            schedule_data.append(entry)
                continue
            # If not a table, try to parse div-based grid
            # Look for column headers
            headers = [el.get_text(strip=True).lower() for el in next_sibling.find_all(['th', 'td', 'div', 'span'])[:10]]
            if any('session' in h or 'presentation' in h for h in headers):
                # Try to parse rows
                rows = next_sibling.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th', 'div', 'span'])
                    if len(cells) >= 5:
                        cell_texts = [c.get_text(strip=True) for c in cells]
                        # Heuristic mapping
                        entry = {
                            'title': '', 'date': current_date, 'time': '', 'type': '', 'contributors': '', 'location': ''
                        }
                        for idx, text in enumerate(cell_texts):
                            t = text.lower()
                            if 'am' in t or 'pm' in t:
                                entry['time'] = text
                            elif 'of a feather' in t or 'talk' in t or 'course' in t or 'labs' in t:
                                entry['type'] = text
                            elif ',' in t and len(t.split(',')) > 1:
                                entry['contributors'] = text
                            elif 'room' in t or 'building' in t:
                                entry['location'] = text
                            elif len(text) > 0 and entry['title'] == '':
                                entry['title'] = text
                        if entry['title']:
                            schedule_data.append(entry)
    # Fallback: try to find all tables in the page
    if not schedule_data:
        print("No schedule found by heading-based parsing. Trying all tables on the page...")
        try:
            dfs = pd.read_html(html)
        except Exception as e:
            print(f"Error parsing tables with pandas: {e}")
            return []
        for df in dfs:
            col_map = {}
            for col in df.columns:
                col_lower = str(col).strip().lower()
                if 'session' in col_lower or 'presentation' in col_lower:
                    col_map['title'] = col
                elif 'time' in col_lower:
                    col_map['time'] = col
                elif 'type' in col_lower:
                    col_map['type'] = col
                elif 'contributor' in col_lower:
                    col_map['contributors'] = col
                elif 'location' in col_lower:
                    col_map['location'] = col
            if all(k in col_map for k in ['title', 'time', 'type', 'contributors', 'location']):
                for _, row in df.iterrows():
                    entry = {
                        'title': row[col_map['title']],
                        'date': '',
                        'time': row[col_map['time']],
                        'type': row[col_map['type']],
                        'contributors': row[col_map['contributors']],
                        'location': row[col_map['location']]
                    }
                    schedule_data.append(entry)
    return schedule_data


def clean_data(data):
    """Clean and validate the extracted data."""
    cleaned = []
    for row in data:
        entry = {k: (str(v).strip() if v is not None else '') for k, v in row.items()}
        # Basic validation: require title and time
        if entry['title'] and entry['time']:
            cleaned.append(entry)
    return cleaned


def save_to_csv(data, filename):
    """Save the data to a CSV file."""
    df = pd.DataFrame(data, columns=REQUIRED_COLUMNS)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")


def main():
    html = fetch_page(URL)
    schedule = parse_schedule(html)
    if not schedule:
        print("No schedule data found matching the required columns.")
        raise SystemExit(1)
    cleaned = clean_data(schedule)
    if not cleaned:
        print("No valid rows after cleaning.")
        raise SystemExit(1)
    save_to_csv(cleaned, 'output.csv')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Script failed: {e}")
