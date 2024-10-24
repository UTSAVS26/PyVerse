# URL Shortener

## 🚩 Aim
To create a simple URL shortener that converts long URLs into shorter, more manageable versions.

## 📝 Description
This is a basic URL shortener implemented in Python. The script takes a long URL as input and generates a short, random URL that can be used to easily reference the original URL. This project serves as an excellent practice for string manipulation, data storage techniques, and can be extended to include a web interface.

### Features:
- Generates a unique shortened URL for each long URL provided.
- Retrieves the original URL when provided with a shortened version.
- Easily expandable into a web application using frameworks like Flask or Django.

## ⚙️ How It Works
1. **URL shortening**: The script generates a random 6-character string using uppercase, lowercase letters, and digits. This string is appended to a base URL (`http://short.url/`) to create the shortened URL.
2. **Storage**: A Python dictionary is used to store the mapping between long URLs and their shortened versions.
3. **URL retrieval**: The original long URL can be retrieved by inputting the shortened URL.

## 🚀 Getting Started

### Prerequisites:
- Python 3.x installed on your machine.

### Installation:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/url-shortener.git
   cd url-shortener

2.