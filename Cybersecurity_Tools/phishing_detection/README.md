# Phishing URL Detector

A machine learning-based system that detects potential phishing URLs by analyzing various URL features. The system uses a Voting Classifier model trained on multiple URL characteristics to determine if a URL is legitimate or potentially malicious.

## Features

- Real-time URL analysis
- 30 different feature extractions including:
  - Domain-based features
  - URL-based features
  - HTML and JavaScript features
  - Domain age and registration features
- User-friendly command-line interface
- Clear warning system for potentially malicious URLs
- Continuous checking capability

## Prerequisites

```bash
pip install joblib numpy requests beautifulsoup4 whois python-whois dnspython tldextract
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/phishing-detector.git
cd phishing-detector
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Ensure all files are in the same directory:
   - `voting_classifier_model.pkl` (trained model)
   - `features_extraction.py` (feature extraction code)
   - `predict.py` (prediction script)

## Usage

1. Run the prediction script:

```bash
python predict.py
```

2. Enter the URL you want to check when prompted:

```
Enter URL to check (or 'quit' to exit): https://example.com
```

3. The system will analyze the URL and provide one of two responses:

   - ✅ This URL appears to be LEGITIMATE
   - ⚠️ Warning: This URL is potentially PHISHING

4. You can choose to check another URL or exit the program.

## How It Works

1. **Feature Extraction**: The system extracts 30 different features from the provided URL using the `features_extraction.py` module, including:

   - IP address presence
   - URL length
   - Shortening service usage
   - '@' symbol presence
   - SSL certificate status
   - Domain age
   - HTML/JavaScript features
   - And many more...

2. **Prediction**: The extracted features are processed by a trained Voting Classifier model that combines multiple machine learning algorithms to make a final prediction.

## Error Handling

The system includes robust error handling for:

- Invalid URLs
- Network connection issues
- Feature extraction failures
- Model loading problems

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. While it can help identify potential phishing URLs, it should not be the sole factor in determining a URL's legitimacy. Always exercise caution when visiting unknown websites.

## Acknowledgments

- Feature extraction methods based on common phishing detection techniques
- Model trained on [dataset reference]
- Thanks to all contributors who participated in this project
