import string
import random

class URLShortener:
    def __init__(self):
        self.url_mapping = {}  # Stores long URLs and their corresponding short URLs
        self.base_url = "http://short.url/"
        self.chars = string.ascii_letters + string.digits  # Characters for generating short URL codes

    def generate_short_code(self, length=6):
        """Generate a random short code for the URL."""
        return ''.join(random.choice(self.chars) for _ in range(length))

    def shorten_url(self, long_url):
        """Shorten a given long URL."""
        if long_url in self.url_mapping:
            return self.url_mapping[long_url]
        else:
            short_code = self.generate_short_code()
            short_url = self.base_url + short_code
            self.url_mapping[long_url] = short_url
            return short_url

    def retrieve_url(self, short_url):
        """Retrieve the original long URL from the shortened version."""
        for long_url, short in self.url_mapping.items():
            if short == short_url:
                return long_url
        return None

# Driver code
if __name__ == "__main__":
    shortener = URLShortener()

    # Example Usage
    print("=== URL Shortener ===")
    long_url = input("Enter the URL to shorten: ")
    short_url = shortener.shorten_url(long_url)
    print(f"Shortened URL: {short_url}")

    # Retrieve original URL
    print("\n=== Retrieve Original URL ===")
    original_url = shortener.retrieve_url(short_url)
    print(f"Original URL for {short_url}: {original_url}")
