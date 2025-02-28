import re
from urllib.parse import urlparse
from Levenshtein import distance as levenshtein_distance

# Define constants for suspicious patterns
PHISHING_KEYWORDS = ["verify your account", "urgent", "immediate action", "login now", "password reset"]
LEGITIMATE_DOMAINS = ["example.com", "service.com"]  # Add more legitimate domains

# Precompile the regular expression for URL extraction
URL_PATTERN = re.compile(r'(https?://\S+)')

# Function to check for suspicious words in the email body
def check_suspicious_words(email_body):
    for keyword in PHISHING_KEYWORDS:
        if keyword.lower() in email_body.lower():
            return True, f"Suspicious word detected: {keyword}"
    return False, ""

# Function to check if the email domain is suspicious
def check_domain(email_sender):
    domain = email_sender.split('@')[-1].lower()
    if domain not in LEGITIMATE_DOMAINS:
        # Optional: Check for domain similarity (e.g., "serv1ce.com" vs. "service.com")
        for legit_domain in LEGITIMATE_DOMAINS:
            if levenshtein_distance(domain, legit_domain) < 3:
                return True, f"Suspicious domain (similar to {legit_domain}): {domain}"
        return True, f"Unknown domain: {domain}"
    return False, ""

# Function to check for suspicious URLs in the email body
def check_url_suspicion(email_body):
    urls = URL_PATTERN.findall(email_body)
    for url in urls:
        parsed_url = urlparse(url)
        # Check for known suspicious patterns like ".xyz" or punycode (e.g., xn--)
        if parsed_url.netloc.endswith('.xyz') or 'xn--' in parsed_url.netloc:
            return True, f"Suspicious URL detected: {url}"
        # Check if the URL uses an IP address instead of a domain name
        if re.match(r'\d{1,3}(\.\d{1,3}){3}', parsed_url.netloc):
            return True, f"IP-based URL detected: {url}"
    return False, ""

# Main function that aggregates the checks
def phishing_detector(email_sender, email_body):
    reasons = []
    
    # Check for suspicious words in the email body
    suspicious_words, reason = check_suspicious_words(email_body)
    if suspicious_words:
        reasons.append(reason)
    
    # Check if the sender's domain is suspicious
    suspicious_domain, reason = check_domain(email_sender)
    if suspicious_domain:
        reasons.append(reason)
    
    # Check for suspicious URLs in the email body
    suspicious_url, reason = check_url_suspicion(email_body)
    if suspicious_url:
        reasons.append(reason)
    
    # Return the result based on the checks
    if reasons:
        return f"Phishing detected due to the following reasons: {', '.join(reasons)}"
    return "No phishing detected."

# Taking user input
email_sender = input("Enter the sender's email address: ")
email_body = input("Enter the email body: ")

# Call the phishing detector function and display the result
result = phishing_detector(email_sender, email_body)
print(result)
