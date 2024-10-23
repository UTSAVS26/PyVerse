import re
import logging
from flask import Flask, request, jsonify
from html import escape
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Rate limiting configuration
limiter = Limiter(app, key_func=get_remote_address)

# Set up logging for security events
logging.basicConfig(filename='security.log', level=logging.INFO)

# Expanded SQL Injection and XSS patterns
SQL_INJECTION_PATTERN = re.compile(
    r"(?:--|;|'|\"|OR|AND|SELECT|INSERT|DELETE|UPDATE|DROP|UNION|#|/\*|\*/|CHAR|HEX)", re.IGNORECASE)
XSS_PATTERN = re.compile(r"(<script.*?>|<.*?on[a-zA-Z]+\s*=|javascript:|data:text/html)", re.IGNORECASE)

def is_safe(input_string):
    """Check if the input string is safe from SQL injection and XSS."""
    if SQL_INJECTION_PATTERN.search(input_string):
        logging.warning(f"SQL Injection detected: {input_string}")
        return False, "Potential SQL Injection detected."
    if XSS_PATTERN.search(input_string):
        logging.warning(f"XSS attack detected: {input_string}")
        return False, "Potential XSS detected."
    return True, ""

@app.route('/submit', methods=['POST'])
@limiter.limit("10 per minute")  # Rate limiting: max 10 requests per minute
def submit():
    try:
        data = request.json
        if not data or "user_input" not in data:
            return jsonify({"error": "Invalid input. Please provide valid JSON."}), 400

        user_input = data.get("user_input", "")
        # Input length validation
        if len(user_input) > 1000:
            return jsonify({"error": "Input too long."}), 413

        # Sanitize the input to escape harmful HTML characters
        safe_input = escape(user_input)

        # Check if the input is safe
        is_safe_input, reason = is_safe(safe_input)
        if not is_safe_input:
            return jsonify({"error": reason}), 400

        # Log and process the safe input
        logging.info(f"Safe input processed: {safe_input}")
        return jsonify({"message": "Input processed successfully!"}), 200

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "An error occurred processing the request."}), 500

if __name__ == '__main__':
    app.run(debug=True)
