## Automation Tool for Web Scraping

### 🎯 Goal

The goal of this project is to access the HTML structure of a particular webpage and extract useful information or data from it using Python. This project focuses on scraping book information from [Books to Scrape](https://books.toscrape.com/), a mock online bookstore website.

### 🧾 Description

This project demonstrates the implementation of web scraping in Python. The script scrapes book titles, prices, and availability status from [Books to Scrape](https://books.toscrape.com/) and saves the extracted data into a CSV file for further analysis.

### 🧮 Features Implemented

- **Scrapes book titles, prices, and availability status** from [Books to Scrape](http://books.toscrape.com/).
- **Iterates through multiple pages** to collect data from the entire catalog.
- **Saves the scraped data** into a structured CSV file for further use or analysis.

### 📚 Libraries Needed

- **BeautifulSoup**: To parse the HTML content and extract useful information.
- **Requests**: To fetch the HTML content from web pages.
- **Pandas**: To store the extracted data and save it into a CSV file.

### 📊 Example Output:

The output CSV file `books_data.csv` will contain data in the following structure:

```csv
Title,Price,Availability
A Light in the Attic,£51.77,In stock
Tipping the Velvet,£53.74,In stock
Soumission,£50.10,In stock
...,...,...
```

**AYESHA NAZNIN**  
[![GitHub](https://img.shields.io/badge/github-%2312100E.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhantomANaz) | [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayesha-naznin-73316a11a/)
