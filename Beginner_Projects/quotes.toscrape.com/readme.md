A simple web scraping program using Python that retrieves data from a website. For this example, we will scrape quotes from [quotes.toscrape.com](http://quotes.toscrape.com/), which is a site specifically designed for practicing web scraping.

### Simple Web Scraping Program

#### Requirements
You'll need to install the `requests` and `BeautifulSoup` libraries. You can do this using pip:

```bash
pip install requests beautifulsoup4
```

### Explanation
1. **Import Libraries**: The program imports the `requests` library to handle HTTP requests and `BeautifulSoup` from `bs4` to parse HTML content.
  
2. **Function Definition**: The `scrape_quotes()` function:
   - Defines the URL of the site to scrape.
   - Sends a GET request to fetch the webpage content.
   - Checks if the response status code is 200 (OK).
   - Parses the HTML content using BeautifulSoup.
   - Finds all quote elements by searching for `div` tags with the class `quote`.
   - Loops through each quote element, extracting the text and the author, and prints them.

3. **Run the Scraper**: The last line calls the `scrape_quotes()` function to execute the scraping process.

### How to Run the Program
1. Ensure you have Python installed on your machine.
2. Install the required libraries as mentioned above.
3. Copy the provided code into a Python file (e.g., `scrape_quotes.py`).
4. Run the script from your terminal or command prompt:

   ```bash
   python scrape_quotes.py
   ```

### Output
The program will print out the quotes and their authors from the specified webpage, like this:

```
Quote: "The world as we have created it is a process of our thinking. It cannot be changed without changing our thinking."
Author: Albert Einstein

Quote: "It is our choices, Harry, that show what we truly are, far more than our abilities."
Author: J.K. Rowling

...
```

### Note
- The example above is designed for educational purposes and uses a public website that allows scraping.
- Always check a website's `robots.txt` file and terms of service to ensure that scraping is allowed.

If you have any questions or need further help with web scraping, feel free to ask!
Here's a simple script that scrapes quotes and their authors from the website: