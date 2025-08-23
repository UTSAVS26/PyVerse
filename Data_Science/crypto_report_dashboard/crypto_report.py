import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import email.encoders
import requests
import pandas as pd
from datetime import datetime
import os


from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the email credentials from environment variables
sender_mail = st.secrets["email"]["EMAIL_USER"]  # Email address



def send_mail(subject, body, filename, sender_mail, receiver_mail):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    email_password = st.secrets["email"]["EMAIL_PASSWORD"]  # Email password # Use environment variable for password

    message = MIMEMultipart()
    message['From'] = sender_mail
    message['To'] = receiver_mail
    message['Subject'] = subject

    # Add body content (including the link to your deployed Streamlit app)
    message.attach(MIMEText(body, 'html'))

    # Attach the crypto data CSV file
    with open(filename, 'rb') as file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(file.read())
        email.encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
        message.attach(part)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_mail, email_password)
            server.sendmail(sender_mail, receiver_mail, message.as_string())
            st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Unable to send mail: {e}")

def get_crypto_data():
    url = 'https://api.coingecko.com/api/v3/coins/markets'
    param = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 250,
        'page': 1
    }
    response = requests.get(url, params=param)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df = df[['id', 'current_price', 'market_cap', 'price_change_percentage_24h', 'high_24h', 'low_24h', 'ath', 'atl']]
        today = datetime.now().strftime('%d-%m-%Y %H-%M-%S')
        df['time_stamp'] = today

        top_negative_10 = df.nsmallest(10, 'price_change_percentage_24h')
        top_positive_10 = df.nlargest(10, 'price_change_percentage_24h')

        file_name = f'crypto_data_{today}.csv'
        df.to_csv(file_name, index=False)

        subject = f"Top 10 Crypto Currency Data to Invest for {today}"

        body = f"""
        <html>
        <body>
        <h2>Good Morning!</h2>
        <p>Your cryptocurrency report is here!</p>
        <h3>Top 10 Cryptos with Highest Price Increase in Last 24 Hours</h3>
        <table border="1">
        <tr><th>Rank</th><th>Crypto Name</th><th>Current Price</th><th>Price Change (%)</th><th>Market Cap</th></tr>
        """
        for idx, row in top_positive_10.iterrows():
            body += f"<tr><td>{idx + 1}</td><td>{row['id']}</td><td>${row['current_price']}</td><td>{row['price_change_percentage_24h']}%</td><td>${row['market_cap']}</td></tr>"
        
        body += f"""
        </table>
        <h3>Top 10 Cryptos with Highest Price Decrease in Last 24 Hours</h3>
        <table border="1">
        <tr><th>Rank</th><th>Crypto Name</th><th>Current Price</th><th>Price Change (%)</th><th>Market Cap</th></tr>
        """
        for idx, row in top_negative_10.iterrows():
            body += f"<tr><td>{idx + 1}</td><td>{row['id']}</td><td>${row['current_price']}</td><td>{row['price_change_percentage_24h']}%</td><td>${row['market_cap']}</td></tr>"
        
        body += f"""
        </table>
        <p>Attached is the full report of over 250 cryptocurrencies.</p>
        <p>Here is the link to your live crypto dashboard: <a href="https://crypto-live-dashboard.streamlit.app/">Crypto Live Dashboard</a></p>
        <p>Regards,<br>Your Crypto Python Application</p>
        </body>
        </html>
        """
        return subject, body, file_name
    else:
        st.error(f"Connection Failed. Error Code {response.status_code}")
        return None, None, None

# Streamlit Interface
st.title("Crypto Report Generator")

email_id = st.text_input("Enter your Gmail Address")
if not email_id:
    st.warning("Please enter a valid email address.")

# Your email address here
sender_mail = st.secrets["email"]["EMAIL_USER"]  # Replace this with your Gmail address

if st.button('Send Report') and email_id:
    # Make sure the email provided is valid
    if '@' not in email_id:
        st.error("Please enter a valid email address.")
    else:
        subject, body, file_name = get_crypto_data()
        if subject and body and file_name:
            send_mail(subject, body, file_name, sender_mail, email_id)
        else:
            st.error("Failed to get crypto data. Please try again later.")
