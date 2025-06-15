import os
import csv
import sys
import argparse
from datetime import datetime

try:
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import smtplib
except ImportError:
    print("Required libraries missing. Make sure you're using Python 3.x.")
    sys.exit(1)

from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------- File paths ----------
CONTACTS_FILE  = BASE_DIR / "contacts.csv"
SENT_FILE      = BASE_DIR / "sent.csv"
TEMPLATE_FILE  = BASE_DIR / "template.html"
# ---------- Load contacts ----------
def load_contacts(file_path):
    if not os.path.exists(file_path):
        print(f"[✗] Missing file: {file_path}")
        sys.exit(1)

    contacts = []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "name" in row and "email" in row and "birthdate" in row:
                contacts.append(row)
            else:
                print("[!] Skipped malformed contact:", row)
    return contacts

# ---------- Load sent list ----------
def load_sent(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'email' not in reader.fieldnames:
            print(f"[!] 'email' column missing in {file_path}. Ignoring file.")
            return set()
        return set(row['email'] for row in reader if 'email' in row)

# ---------- Save sent email (fixed version) ----------
def save_sent(email):
    header = ["email", "date"]
    need_header = True

    if os.path.exists(SENT_FILE):
        with open(SENT_FILE, "r", encoding="utf-8") as f:
            first_line = f.readline().strip().lower()
            if first_line == ",".join(header):
                need_header = False

    with open(SENT_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow([email, datetime.now().date()])

# ---------- Load template ----------
def load_template(file_path):
    if not os.path.exists(file_path):
        print(f"[✗] Missing file: {file_path}")
        sys.exit(1)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ---------- Send email ----------
def send_email(to_email, subject, html_content, sender_email, sender_password):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    part = MIMEText(html_content, "html")
    msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        print(f"[✓] Sent email to {to_email}")
    except Exception as e:
        print(f"[✗] Failed to send email to {to_email}: {e}")

# ---------- Validate credentials ----------
def validate_credentials(email, password):
    if not email or not password:
        print("[✗] Email and password must be provided.")
        sys.exit(1)
    if "@" not in email:
        print("[✗] Invalid email format.")
        sys.exit(1)

# ---------- Main bot logic ----------
def run_bot(sender_email, sender_password):
    validate_credentials(sender_email, sender_password)

    today = datetime.now().strftime("%m-%d")
    contacts = load_contacts(CONTACTS_FILE)
    sent_emails = load_sent(SENT_FILE)
    template = load_template(TEMPLATE_FILE)

    for contact in contacts:
        email = contact["email"].strip()
        name = contact["name"].strip()
        birthdate = contact["birthdate"].strip()

        if not email or not name or not birthdate:
            continue

        if email in sent_emails:
            continue

        try:
            contact_date = datetime.strptime(birthdate, "%Y-%m-%d").strftime("%m-%d")
        except ValueError:
            print(f"[!] Invalid birthdate format for {email}. Skipping.")
            continue

        if contact_date == today:
            personalized_html = template.replace("{{name}}", name)
            subject = f"🎉 Happy Birthday, {name}!"
            send_email(email, subject, personalized_html, sender_email, sender_password)
            save_sent(email)

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎂 Auto Birthday Greeting Bot")
    parser.add_argument("email", help="Sender email address (e.g., your@gmail.com)")
    parser.add_argument("password", help="Sender email password or app password")
    args = parser.parse_args()
    run_bot(args.email, args.password)
