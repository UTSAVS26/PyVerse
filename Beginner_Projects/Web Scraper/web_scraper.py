import requests
from bs4 import BeautifulSoup

# Function to scrape article titles from a specified website
def scrape_titles(url):
    # Set headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }

    # Send a GET request to the website
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    print("HTTP Status Code:", response.status_code)  # Debugging print
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all headings (h2, h3, etc.) - you can modify the tag selection here
        titles = soup.find_all(['h2', 'h3'])

        # Extract and print the titles
        print("Article Titles from", url)
        found_titles = False
        for idx, title in enumerate(titles):
            text = title.get_text()
            if text:  # Check if there's text in the heading
                print(f"{idx + 1}. {text.strip()}")
                found_titles = True

        if not found_titles:
            print("No article titles found.")
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)

# Main block to get URL input from the user
if __name__ == "__main__":
    website_url = input("Enter the URL of the website to scrape: ")
    scrape_titles(website_url)
