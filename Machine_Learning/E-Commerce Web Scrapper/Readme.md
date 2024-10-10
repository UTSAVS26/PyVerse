
# PROJECT TITLE

E-Commerce Website Web Scrapper using Python implemented in Jupyter Notebook

## GOAL

**Aim**: To scrape product related data using Python libraries for web scrapping like BeautifulSoup.

## DESCRIPTION

A web scraper for an e-commerce website extracts product details such as name, price, description, ratings, and reviews of a specific product. It automates data collection, helping users track price changes, reviews, and competitor analysis efficiently.

## Steps How Web Scrapping Works

1. We first import required libraries in our ipynb file.
2. Then we extract **URL** of the required product from any e-commerce website like **Flipkart** and **Amazon**.
3. Using **requests** library we request HTML code from the website.
4. To find the particular element we use **inspect** feature provided by Browser to know the class name of **div**.
5. Using that class name we find that particular detail like Name or Price of product.
6. In this way we can repeat same process as in point no.5 to find other info. related to product. 

## LIBRARIES USED

- **requests**: Python library used to send HTTP requests to interact with web services and retrieve data from the internet. It simplifies making GET, POST, PUT, DELETE, and other HTTP requests.
- **BeautifulSoup**: Python library used for web scraping purposes to parse HTML and XML documents.