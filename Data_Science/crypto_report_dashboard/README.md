

# Crypto Report Generator and Dashboard

Welcome to the **Crypto Report Generator and Dashboard**! This project allows users to generate cryptocurrency reports via email, with a clean and interactive interface built using Streamlit. It fetches real-time crypto data and sends detailed reports via email with a CSV attachment.

Additionally, the project includes a live crypto dashboard that visualizes this data, offering insights into the top 10 cryptocurrencies based on their 24-hour price change.

## Features

- **Real-time Crypto Data:** Fetches the latest market data for over 250 cryptocurrencies using the CoinGecko API.
- **Report Generation:** Generates a detailed cryptocurrency report with the top 10 cryptocurrencies that have seen the highest price increase and decrease in the last 24 hours.
- **Email Integration:** Sends the generated report via email with the option to attach the full data in CSV format.
- **User Input:** Collects the recipient's email address for sending reports.

## Technologies Used

- **Python:** The main programming language used for backend logic.
- **Streamlit:** For creating the interactive web interface.
- **pandas:** For data manipulation and report generation.
- **requests:** For fetching cryptocurrency data from the CoinGecko API.
- **smtplib:** For sending emails with reports and attachments.
- **dotenv:** (optional in local setup) To securely store and manage API keys and email credentials.
- **HTML:** For formatting the email body with a rich structure.

## Installation

### Prerequisites

To get started with the project, you need to have Python installed on your local machine.

### Clone the repository:

### Create a virtual environment:

```bash
python -m venv venv
```

### Activate the virtual environment:

For **Windows**:

```bash
venv\Scripts\activate
```

For **macOS/Linux**:

```bash
source venv/bin/activate
```

### Install required dependencies:

You don’t need to create a `.env` file since the credentials are handled securely via Streamlit secrets. If you still want to test locally, you can use the `.streamlit/secrets.toml` file for credentials.

```bash
pip install -r requirements.txt
```

### Run the Streamlit app:

```bash
streamlit run crypto_report.py
```

This will launch the Streamlit app in your default browser.

## How It Works

1. **User Input:** The user provides their email address in the Streamlit UI.
2. **Data Fetching:** The app fetches real-time cryptocurrency data from the CoinGecko API.
3. **Report Generation:** The app generates a CSV report that includes the top 10 cryptocurrencies with the highest price change (both positive and negative) in the last 24 hours.
4. **Sending Email:** The user can click the "Send Report" button, which triggers the sending of the report via email (with CSV attachment) to the provided email address.

## Features Overview

- **Top 10 Cryptos with Highest Price Increase:** Displays the top 10 cryptocurrencies that have seen the highest increase in their price in the last 24 hours.
- **Top 10 Cryptos with Highest Price Decrease:** Displays the top 10 cryptocurrencies that have seen the largest decrease in their price in the last 24 hours.
- **Crypto Dashboard:** A live crypto dashboard provides a real-time view of cryptocurrency market data. Check it out [here](https://crypto-live-dashboard.streamlit.app/).

## File Structure

```text
crypto_report_dashboard/
│
├── crypto_report.py        # Main Streamlit app to run the UI
├── .streamlit/             # Contains Streamlit secrets and config files
│   └── secrets.toml        # Email credentials for Streamlit deployment
├── requirements.txt        # Python dependencies             
├── README.md               # This readme file
```

## Contributing

1. Fork the repository
2. Clone your fork locally
   
```bash
git clone https://github.com/UTSAVS26/PyVerse.git
cd PyVerse/Data_Science/crypto_report_dashboard
```
4. Create a new branch (`git checkout -b feature-branch`)
5. Make your changes and commit them (`git commit -am 'Add feature'`)
6. Push your changes to your fork (`git push origin feature-branch`)
7. Create a new Pull Request

## Configuration

Create a file at `.streamlit/secrets.toml` with your email credentials:

```toml
[email]
EMAIL_USER     = "your_email@gmail.com"       # Gmail address used to send the report
EMAIL_PASSWORD = "your_app_password"          # Your Gmail app password or account password
```

## Acknowledgments

- **CoinGecko API:** For providing free access to cryptocurrency data.
- **Streamlit:** For providing a simple and powerful tool for building interactive web apps.
- **Python libraries:** pandas, requests, smtplib, and others that helped build this project.

---
<img width="1919" height="888" alt="Screenshot 2025-08-15 163051" src="https://github.com/user-attachments/assets/06f0c1a0-c823-43ac-81ad-1906ef895297" />
<img width="1919" height="887" alt="Screenshot 2025-08-15 163021" src="https://github.com/user-attachments/assets/36aef11a-0568-4e8d-b40e-8a05639f0c02" />



