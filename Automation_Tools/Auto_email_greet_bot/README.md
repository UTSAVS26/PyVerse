# 🎂 Auto Birthday Greeting Bot (Email Edition)

An automated Python bot that sends beautiful HTML email greetings on birthdays using contact details from a CSV file. Perfect for staying thoughtful — without having to remember every birthday!

---

## ✨ Features

- ✅ Sends **custom HTML email greetings**
- ✅ Reads contacts from `contacts.csv`
- ✅ Checks if **today matches a contact’s birthday**
- ✅ Keeps track of emails already sent using `sent.csv`
- ✅ Prevents duplicate greetings
- ✅ Uses a customizable `template.html`
- ✅ CLI support for easy manual execution
- ✅ Fully documented and production-ready

---

## 📁 Folder Structure

```
AutoGreetingBot/
│
├── auto_greet_bot.py       # Main Python script
├── contacts.csv            # Contact list (name, email, birthdate)
├── sent.csv                # Log of already-greeted contacts (auto-generated)
├── template.html           # Custom HTML email template
└── README.md               # This file
```

---

## 🛠 Requirements

- Python 3.6+
- Internet access (to send emails)
- Gmail or SMTP-compatible account with **App Password**

---

## 🐍 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/auto-birthday-greeting-bot.git
   cd auto-birthday-greeting-bot
   ```
2. **Install required libraries** (usually built-in):

   ```bash
   pip install --upgrade pip
   ```
3. **Set up your contacts and template:**

   * Fill `contacts.csv` with:

     ```csv
     name,email,birthdate
     Alice,alice@example.com,1995-06-15
     Bob,bob@example.com,1990-12-01
     ```
   * Customize `template.html` as needed.

---

## 🔐 Gmail App Password Setup

If you're using Gmail:

1. Enable 2-Step Verification:[https://myaccount.google.com/security](https://myaccount.google.com/security)
2. Generate an App Password:[https://myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Use that app password when running the script.

---

## 🚀 Usage

Run the bot manually via CLI:

```bash
python auto_greet_bot.py your_email@gmail.com your_app_password
```

Example:

```bash
python auto_greet_bot.py alice.sender@gmail.com abcd1234efgh5678
```

✅ The bot will:

- Check today’s date
- Match it against birthdays in `contacts.csv`
- Send customized greetings
- Log sent emails in `sent.csv`

---

## 📄 File Formats

### `contacts.csv`

```csv
name,email,birthdate
John,john@example.com,1990-06-15
```

- `birthdate` format: `YYYY-MM-DD`

### `template.html`

```html
<!DOCTYPE html>
<html>
  <head><title>Happy Birthday!</title></head>
  <body>
    <h2>Happy Birthday, {{name}}!</h2>
    <p>Have a wonderful day 🎉</p>
  </body>
</html>
```

- Replace `{{name}}` with the actual contact's name dynamically.

---

## 📌 Notes

- Do **not** leave `sent.csv` open in Excel while running the bot.
- The bot won’t resend greetings for the same person on the same day.
- Handles invalid rows gracefully.

---

## 👤 Author

**Shivansh Katiyar**
GitHub: [SK8-infi](https://github.com/SK8-infi)

---
