
# LinkedIn Job Scraper

A web scraper designed to extract job postings from LinkedIn using Scrapy and Octoparse proxy middleware. This project addresses challenges related to scraping job data while minimizing the risk of getting blocked.

## Features
- Scrapes job postings from LinkedIn with relevant details including job title, company name, location, and job description.
- Utilizes Octoparse proxy middleware for IP rotation, reducing the chances of getting blocked.
- Implements error handling and logging for monitoring scraping performance.
- Supports dynamic content scraping to capture all necessary job details.

  
## Usage
To run the scraper and start extracting job postings:
1. **Ensure your proxy settings are correctly configured in `settings.py`.**
2. **Start the Scrapy spider:**
   ```bash
   scrapy crawl linkedin_jobs
   ```
3. **The scraped job postings will be stored in the specified format (e.g., JSON, CSV) as defined in the Scrapy settings.** 

## Configuration
- **Proxy Middleware**: The Octoparse proxy middleware is set up in the Scrapy settings. Make sure to configure your proxy settings as needed in `settings.py`.
- **CSS Selectors**: The CSS selectors used for extracting job details are customizable in the spider file. Adjust them if necessary to match changes in the LinkedIn layout.

## Running the Scraper
### Running in Different Modes
- To run the spider and save the output to a JSON file:
  ```bash
  scrapy crawl linkedin_jobs -o output.json
  ```
- To run the spider and save the output to a CSV file:
  ```bash
  scrapy crawl linkedin_jobs -o output.csv
  ```

### Checking Logs
- Logs are generated to track the scraping process. You can check the log file to troubleshoot any issues. The log level can be adjusted in `settings.py`.

## Additional Commands
- **List all available commands**:
  ```bash
  scrapy
  ```
- **Run the spider with a specific setting**:
  ```bash
  scrapy crawl linkedin_jobs -s LOG_LEVEL=INFO
  ```
This provides a single, cohesive README document that includes all the necessary information about your LinkedIn scraper project.
