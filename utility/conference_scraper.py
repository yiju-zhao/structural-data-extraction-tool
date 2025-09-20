import csv
import argparse
import time
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def scrape_conference_data(url, output_file):
    """
    Scrapes conference data from a given URL and saves it to a CSV file.
    Uses Selenium to handle JavaScript-rendered content.
    """
    html_content = None
    try:
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        print("Setting up browser driver...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        print(f"Fetching URL: {url}...")
        driver.get(url)

        # Wait for JavaScript to load content. Prefer explicit waits over fixed sleep.
        print("Waiting for dynamic content to load...")
        try:
            # Wait until event cards are present
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'div.displaycards.touchup-date'))
            )
            # Then wait until at least one time/date element is present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[class*="touchup-date-div"], [id^="touchup-date-"]'))
            )
        except Exception:
            # Fallback to a short sleep if waits time out
            time.sleep(3)

        html_content = driver.page_source
        print("Page source fetched successfully.")

    except WebDriverException as e:
        print(f"A WebDriver error occurred: {e}")
        print("Please ensure you have Google Chrome installed and an internet connection.")
        return
    except Exception as e:
        print(f"An unexpected error occurred with Selenium: {e}")
        return
    finally:
        if 'driver' in locals() and driver:
            driver.quit()

    if not html_content:
        print("Failed to fetch page content.")
        return

    soup = BeautifulSoup(html_content, 'html.parser')
    # Robust selection: tolerate class order and require an id that starts with 'event-'
    events = soup.select('div.displaycards.touchup-date[id^="event-"]')
    if not events:
        # Fallback to a more permissive search
        events = [
            div for div in soup.find_all('div', id=re.compile(r'^event-'))
            if div.get('class') and 'displaycards' in div.get('class') and 'touchup-date' in div.get('class')
        ]

    if not events:
        print("No events found with the specified class. The website structure may have changed.")
        return

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Authors', 'Date/Time', 'Abstract']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            title_element = event.find('a', class_='small-title')
            title = title_element.text.strip() if title_element else 'N/A'

            authors_element = event.find('div', class_='author-str')
            authors = authors_element.text.strip() if authors_element else 'N/A'

            # Date/Time: be permissive — search within the card for any descendant
            # whose class contains 'touchup-date-div' or id starts with 'touchup-date-'
            datetime = 'N/A'
            time_el = (
                event.select_one('[class*="touchup-date-div"]')
                or event.find(attrs={"id": re.compile(r'^touchup-date-')})
            )
            if time_el:
                datetime = time_el.get_text(" ", strip=True)

            abstract_element = event.find('details')
            abstract = 'N/A'
            if abstract_element:
                abstract_div = abstract_element.find('div', class_='text-start p-4')
                if abstract_div:
                    abstract = abstract_div.text.strip()

            writer.writerow({
                'Title': title,
                'Authors': authors,
                'Date/Time': datetime,
                'Abstract': abstract
            })

    print(f"Scraping complete. Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape conference data from a website.")
    parser.add_argument("url", help="The URL of the conference website to scrape.")
    parser.add_argument("-o", "--output", default="conference_data.csv", help="The name of the output CSV file.")
    args = parser.parse_args()

    scrape_conference_data(args.url, args.output)
