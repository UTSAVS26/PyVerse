import re
import socket
import ssl
import whois
import requests
from datetime import datetime
from urllib.parse import urlparse, urlencode
from bs4 import BeautifulSoup
import dns.resolver
import tldextract

class URLFeatureExtractor:
    def __init__(self, url):
        self.url = url
        self.domain = urlparse(url).netloc
        self.features = {}
        
    def extract_all_features(self):
        """Extract all features from the URL"""
        self.check_ip_address()
        self.check_url_length()
        self.check_shortening_service()
        self.check_at_symbol()
        self.check_double_slash_redirect()
        self.check_prefix_suffix()
        self.check_sub_domains()
        self.check_ssl()
        self.check_domain_registration()
        self.check_favicon()
        self.check_port()
        self.check_https_token()
        self.check_request_url()
        self.check_url_anchor()
        self.check_links_in_tags()
        self.check_sfh()
        self.check_email_submission()
        self.check_abnormal_url()
        self.check_redirect()
        self.check_mouseover()
        self.check_right_click()
        self.check_popup()
        self.check_iframe()
        self.check_age_of_domain()
        self.check_dns_record()
        self.check_web_traffic()
        self.check_page_rank()
        self.check_google_index()
        self.check_pointing_links()
        self.check_statistical_report()
        
        return self.features

    def check_ip_address(self):
        """Check if URL contains IP address"""
        ip_pattern = re.compile(
            r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        )
        self.features['having_IP'] = 1 if ip_pattern.search(self.url) else -1

    def check_url_length(self):
        """Check URL length"""
        self.features['URL_Length'] = 1 if len(self.url) > 54 else -1

    def check_shortening_service(self):
        """Check if URL is using shortening service"""
        shortening_services = ['bit.ly', 'goo.gl', 't.co', 'tinyurl.com']
        self.features['Shortining_Service'] = 1 if any(service in self.url for service in shortening_services) else -1

    def check_at_symbol(self):
        """Check for @ symbol in URL"""
        self.features['having_At_Symbol'] = 1 if '@' in self.url else -1

    def check_double_slash_redirect(self):
        """Check for double slash redirect"""
        self.features['double_slash_redirecting'] = 1 if '//' in self.url[8:] else -1

    def check_prefix_suffix(self):
        """Check for prefix or suffix separated by dash"""
        self.features['Prefix_Suffix'] = 1 if '-' in self.domain else -1

    def check_sub_domains(self):
        """Check number of subdomains"""
        ext = tldextract.extract(self.url)
        subdomain = ext.subdomain.split('.')
        self.features['having_Sub_Domain'] = 1 if len(subdomain) > 1 else -1

    def check_ssl(self):
        """Check SSL final state"""
        try:
            ctx = ssl.create_default_context()
            with ctx.wrap_socket(socket.socket(), server_hostname=self.domain) as s:
                s.connect((self.domain, 443))
                self.features['SSLfinal_State'] = 1
        except:
            self.features['SSLfinal_State'] = -1

    def check_domain_registration(self):
        """Check domain registration length"""
        try:
            w = whois.whois(self.domain)
            if isinstance(w.expiration_date, list):
                exp_date = w.expiration_date[0]
            else:
                exp_date = w.expiration_date
            
            if exp_date:
                length = (exp_date - datetime.now()).days
                self.features['Domain_registeration_length'] = 1 if length > 365 else -1
            else:
                self.features['Domain_registeration_length'] = -1
        except:
            self.features['Domain_registeration_length'] = -1

    def check_favicon(self):
        """Check favicon"""
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            favicon = soup.find('link', rel='shortcut icon') or soup.find('link', rel='icon')
            self.features['Favicon'] = 1 if favicon and self.domain in favicon.get('href', '') else -1
        except:
            self.features['Favicon'] = -1

    def check_port(self):
        """Check for non-standard port"""
        try:
            parsed = urlparse(self.url)
            self.features['port'] = 1 if parsed.port and parsed.port not in [80, 443] else -1
        except:
            self.features['port'] = -1

    def check_https_token(self):
        """Check HTTPS token in domain"""
        self.features['HTTPS_token'] = 1 if 'https' in self.domain else -1

    def check_request_url(self):
        """Check request URL"""
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            imgs = soup.find_all('img', src=True)
            total_urls = len(imgs)
            suspicious_urls = sum(1 for img in imgs if self.domain not in img['src'])
            self.features['Request_URL'] = 1 if suspicious_urls / total_urls < 0.22 else -1
        except:
            self.features['Request_URL'] = -1

    def check_url_anchor(self):
        """Check URL of anchor"""
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            anchors = soup.find_all('a', href=True)
            total_anchors = len(anchors)
            suspicious_anchors = sum(1 for a in anchors if self.domain not in a['href'])
            self.features['URL_of_Anchor'] = 1 if suspicious_anchors / total_anchors < 0.31 else -1
        except:
            self.features['URL_of_Anchor'] = -1

    def check_links_in_tags(self):
        """Check links in tags"""
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            tags = soup.find_all(['meta', 'script', 'link'])
            total_tags = len(tags)
            suspicious_tags = sum(1 for tag in tags if tag.get('src', '') and self.domain not in tag['src'])
            self.features['Links_in_tags'] = 1 if suspicious_tags / total_tags < 0.17 else -1
        except:
            self.features['Links_in_tags'] = -1

    def check_sfh(self):
        """Check server form handler"""
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            forms = soup.find_all('form', action=True)
            self.features['SFH'] = 1 if any(not form['action'] or 'about:blank' in form['action'] for form in forms) else -1
        except:
            self.features['SFH'] = -1

    def check_email_submission(self):
        """Check for email submission"""
        try:
            response = requests.get(self.url)
            self.features['Submitting_to_email'] = 1 if 'mailto:' in response.text else -1
        except:
            self.features['Submitting_to_email'] = -1

    def check_abnormal_url(self):
        """Check abnormal URL"""
        try:
            w = whois.whois(self.domain)
            self.features['Abnormal_URL'] = 1 if not w.domain_name else -1
        except:
            self.features['Abnormal_URL'] = 1

    def check_redirect(self):
        """Check redirect"""
        try:
            response = requests.get(self.url, allow_redirects=False)
            self.features['Redirect'] = 1 if response.status_code in [301, 302] else -1
        except:
            self.features['Redirect'] = -1

    def check_mouseover(self):
        """Check onMouseOver"""
        try:
            response = requests.get(self.url)
            self.features['on_mouseover'] = 1 if 'onmouseover="window.status' in response.text.lower() else -1
        except:
            self.features['on_mouseover'] = -1

    def check_right_click(self):
        """Check RightClick disabled"""
        try:
            response = requests.get(self.url)
            self.features['RightClick'] = 1 if 'preventdefault()' in response.text.lower() or 'event.button==2' in response.text.lower() else -1
        except:
            self.features['RightClick'] = -1

    def check_popup(self):
        """Check popup window"""
        try:
            response = requests.get(self.url)
            self.features['popUpWidnow'] = 1 if 'window.open' in response.text.lower() else -1
        except:
            self.features['popUpWidnow'] = -1

    def check_iframe(self):
        """Check iframe presence"""
        try:
            response = requests.get(self.url)
            soup = BeautifulSoup(response.text, 'html.parser')
            self.features['Iframe'] = 1 if soup.find_all('iframe') else -1
        except:
            self.features['Iframe'] = -1

    def check_age_of_domain(self):
        """Check age of domain"""
        try:
            w = whois.whois(self.domain)
            if isinstance(w.creation_date, list):
                age = (datetime.now() - w.creation_date[0]).days
            else:
                age = (datetime.now() - w.creation_date).days
            self.features['age_of_domain'] = 1 if age > 180 else -1
        except:
            self.features['age_of_domain'] = -1

    def check_dns_record(self):
        """Check DNS record"""
        try:
            dns.resolver.resolve(self.domain, 'A')
            self.features['DNSRecord'] = 1
        except:
            self.features['DNSRecord'] = -1

    def check_web_traffic(self):
        """Check web traffic (simplified)"""
        try:
            response = requests.get(f"https://data.alexa.com/data?cli=10&url={self.domain}")
            self.features['web_traffic'] = 1 if response.status_code == 200 else -1
        except:
            self.features['web_traffic'] = -1

    def check_page_rank(self):
        """Check page rank (simplified)"""
        self.features['Page_Rank'] = -1  # Actual PageRank API is not freely available

    def check_google_index(self):
        """Check Google index"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            search_url = f"https://www.google.com/search?q=site:{self.domain}"
            response = requests.get(search_url, headers=headers)
            self.features['Google_Index'] = 1 if "did not match any documents" not in response.text else -1
        except:
            self.features['Google_Index'] = -1

    def check_pointing_links(self):
        """Check links pointing to page (simplified)"""
        self.features['Links_pointing_to_page'] = -1  # Would require backlink API service

    def check_statistical_report(self):
        """Check statistical report (simplified)"""
        try:
            response = requests.get(f"https://www.phishtank.com/search.php?query={self.domain}")
            self.features['Statistical_report'] = 1 if response.status_code == 200 else -1
        except:
            self.features['Statistical_report'] = -1

# Example usage
def extract_features(url):
    extractor = URLFeatureExtractor(url)
    features = extractor.extract_all_features()
    return features

if __name__ == "__main__":
    # Example URL
    url = "https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/"
    features = extract_features(url)

    for feature, value in features.items():
        print(f"{feature}: {value}")
    print(len(features))