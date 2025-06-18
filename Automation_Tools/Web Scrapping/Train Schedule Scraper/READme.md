# 🚆 Goibibo Train Schedule Scraper

This project is a **Streamlit-based web app** that uses **Selenium automation** to scrape train details (train name, timings, and prices) from the Goibibo Trains page based on user input.

---

## 📌 Features

- 🔎 **Search Trains** by selecting `From`, `To`, and `Date` easily via the UI.
- 🧠 **Automated Scraping** of Goibibo using Selenium – just like a human user.
- 📋 **Displays Train Info**: train number, name, arrival, departure, source, destination, and cheapest ticket price.
- 💾 **JSON Export**: Saves extracted data to a `train_data_with_prices.json` file.
- 📥 **Download Option**: Lets you download the data from the UI.
- 💡 **Handles Popups** and missing data gracefully with logs and messages.

---

## ⚙️ Setup Instructions

Follow the steps below to run the project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-FORKED-REPO-NAME.git
cd YOUR-FORKED-REPO-NAME
````

### 2️⃣ Create a Virtual Environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the App

```bash
streamlit run fetch_train_schedules.py
```

Then open the provided local URL (e.g., `http://localhost:8501`) in your browser.

---

## 📂 Project Structure

```bash
├── fetch_train_schedules.py   # Main Streamlit app
├── requirements.txt           # Required libraries
└── train_data_with_prices.json (generated after scraping)
```

---

## Implementation Results

![image](https://github.com/user-attachments/assets/0efcaeca-3c50-4bee-9f1d-4c5a1c91a11f)
![image](https://github.com/user-attachments/assets/efb5f55e-9ca2-403b-9b28-ca65ec98cd15)
![image](https://github.com/user-attachments/assets/491d542b-ef06-48b6-b5bb-c727a6328d69)

---
## 💻 Connect With Me

* 👨‍💻 GitHub: [Tanishq-789](https://github.com/Tanishq-789)
* 🔗 LinkedIn: [Tanishq Shinde](https://www.linkedin.com/in/tanishq-shinde977)

---
