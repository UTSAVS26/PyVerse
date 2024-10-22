import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import logging

# Configure logging
logging.basicConfig(filename='scraper.log', level=logging.INFO)

def scrape_static_page(url):
    """Scrape data from a static web page."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract data (example: product prices)
        products = []
        for product in soup.find_all('div', class_='product'):
            name = product.find('h2').text if product.find('h2') else 'N/A'
            price = product.find('span', class_='price').text if product.find('span', class_='price') else 'N/A'
            products.append((name, price))
            logging.info(f"Scraped {name}: {price}")

        return products

    except Exception as e:
        logging.error(f"Error scraping static page: {e}")
        return []

def scrape_dynamic_page(url):
    """Scrape data from a dynamic web page using Selenium."""
    try:
        options = Options()
        options.add_argument('--headless')  # Run in headless mode
        service = Service('/usr/local/bin/chromedriver')  # Path for GitHub Codespaces
        driver = webdriver.Chrome(service=service, options=options)

        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Extract data (example: product prices)
        products = []
        for product in driver.find_elements(By.CLASS_NAME, 'product'):
            name = product.find_element(By.TAG_NAME, 'h2').text if product.find_element(By.TAG_NAME, 'h2') else 'N/A'
            price = product.find_element(By.CLASS_NAME, 'price').text if product.find_element(By.CLASS_NAME, 'price') else 'N/A'
            products.append((name, price))
            logging.info(f"Scraped {name}: {price}")

        driver.quit()
        return products

    except Exception as e:
        logging.error(f"Error scraping dynamic page: {e}")
        return []

if __name__ == "__main__":
    static_url = 'https://example.com/static'
    dynamic_url = 'https://example.com/dynamic'

    static_data = scrape_static_page(static_url)
    dynamic_data = scrape_dynamic_page(dynamic_url)

    print("Static Data:", static_data)
    print("Dynamic Data:", dynamic_data)