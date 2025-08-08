# 🛒 Amazon Product Availability Checker

This is a Python script that checks whether a product on Amazon is **in stock** or **out of stock** by scraping the product page using `requests` and `BeautifulSoup`.

---

## ✅ Features

- Fetches **product title** and **availability status**
- Simple script – beginner-friendly
- Uses `User-Agent` header to mimic browser requests

---

## 📦 Requirements

- Python 3.6 
- libraries like requests,BeautifulSoup
---

## 🧠 How It Works

The script performs the following steps:
1. Sends an HTTP GET request to the provided Amazon product URL
2. Parses the HTML content using BeautifulSoup
3. Extracts the product title and availability info
4. Prints whether the product is in stock or not

---

## 🚀 Usage

1. Clone the repo or download the script.

2. Open `amazon.py` and replace the product URL placeholder with your own:
   ```python
   product_url = "https://www.amazon.in/dp/B0C7BGGRB6"  # or replace the placeholder in the file
   ```

3. Run the script:
   ```bash
   python amazon.py
   ```

4. Example output:
   ```
   Product: Apple iPhone 14 (128 GB) - Midnight
   Availability: In stock.
   ```

---

##  Author

**Harshi Gupta**  

---