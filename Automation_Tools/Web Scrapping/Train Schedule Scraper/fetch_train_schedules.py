import streamlit as st
import time
import json
import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Function to initialize the Selenium driver
def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")   # renders without X server
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager(cache_valid_range=30).install())  # cache 30 days
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# Function to close login/signup pop-up
def close_popup(driver):
    try:
        close = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "/html/body/div/div/div[2]/div[1]"))
        )
        close.click()
-    except:
-        pass
+    except TimeoutException:
+        # Popup did not appear – nothing to close.
+        pass
def enter_from_location(driver, location):
    try:
        from_box = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="__next"]/div/div[2]/div[3]/div[2]/div/div[1]/div[1]/div/p[2]'))
        )
        from_box.click()
        time.sleep(1)

        from_input = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="__next"]/div/div[2]/div[3]/div[2]/div/div[1]/div[1]/div[2]/div/input'))
        )
        from_input.send_keys(location)
        time.sleep(2)

        suggestion = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="__next"]/div/div[2]/div[3]/div[2]/div/div[1]/div[1]/div[2]/ul/li[1]/div/div/p[1]'))
        )
        suggestion.click()
    except Exception as e:
        st.error(f"❌ Error setting 'From' location: {e}")

def enter_to_location(driver, location):
    try:
        to_box = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="__next"]/div/div[2]/div[3]/div[2]/div/div[1]/div[2]/div/p[2]'))
        )
        to_box.click()
        time.sleep(1)

        to_input = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="__next"]/div/div[2]/div[3]/div[2]/div/div[1]/div[2]/div[2]/div/input'))
        )
        to_input.send_keys(location)
        time.sleep(2)

        suggestion = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="__next"]/div/div[2]/div[3]/div[2]/div/div[1]/div[2]/div[2]/ul/li[1]/div/div'))
        )
        suggestion.click()
    except Exception as e:
        st.error(f"❌ Error setting 'To' location: {e}")

def select_date(driver, day):
    try:
        date_box = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="__next"]/div/div[2]/div[3]/div[2]/div/div[1]/div[3]/div/p[2]'))
        )
        date_box.click()
        time.sleep(2)
        date_btn = driver.find_element(By.XPATH, f'//p[text()="{day}"]')
        driver.execute_script("arguments[0].click();", date_btn)
        time.sleep(1)
    except Exception as e:
        st.error(f"❌ Error selecting date: {e}")

def click_search(driver):
    try:
        search = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "span.styles_FswSearchCta__Tf7s4"))
        )
        driver.execute_script("arguments[0].click();", search)
        time.sleep(10)
    except Exception as e:
        st.error(f"❌ Error clicking search button: {e}")

def fetch_train_data(driver):
    try:
        trains = driver.find_elements(By.CSS_SELECTOR, '.TrainCard_trnCrd__wb4xP')
        result = []
        for train in trains[:5]:
            try:
                name = train.find_element(By.CSS_SELECTOR, 'p.rubikSemiBold.font22.blackText2.appendLeft18').text
                no = train.find_element(By.CSS_SELECTOR, 'p.TrainCard_trainCrd_Number__riIV3').text
                arrival = train.find_element(By.CSS_SELECTOR, 'div.makeFlex.row.spaceBetween.hrtlCenter.appendTop16 > div:nth-child(1) > p').text
                from_loc = train.find_element(By.CSS_SELECTOR, 'p.font16.grayText7.appendLeft6.rubik400').text
                departure = train.find_element(By.CSS_SELECTOR, 'div:nth-child(3) > p').text
                to_loc = train.find_element(By.CSS_SELECTOR, 'p.font16.grayText7.appendLeft6.textRight.rubik400').text
                prices = train.find_elements(By.CSS_SELECTOR, 'p.font16.blackText2.rubik400')
                price_vals = []
                for p in prices:
                    txt = p.text.replace("₹", "").strip()
                    if txt.isdigit():
                        price_vals.append(int(txt))
                min_price = f"₹ {min(price_vals)}" if price_vals else "N/A"
                result.append({
                    "train_no": no,
                    "train_name": name,
                    "arrival_time": arrival,
                    "from": from_loc,
                    "departure_time": departure,
                    "to": to_loc,
                    "price": min_price
                })
            except Exception as e:
                st.warning(f"⚠️ Skipping one train due to error: {e}")
        return result
    except Exception as e:
        st.error(f"❌ Could not fetch train results: {e}")
        return []

# Streamlit UI
st.set_page_config(page_title="Train Search", layout="centered")
st.title("🚆 Train Fare Finder (Goibibo Scraper)")
st.markdown("Search for trains by entering From, To, and Date below.")

from_city = st.text_input("From", placeholder="e.g., Mumbai")
to_city = st.text_input("To", placeholder="e.g., Delhi")
date = st.date_input("Travel Date", min_value=datetime.date.today())

if st.button("🔍 Search Trains"):
    if not from_city or not to_city:
        st.warning("Please enter both 'From' and 'To' locations.")
    else:
        with st.spinner("Launching browser and fetching train data..."):
            driver = init_driver()
            try:
                driver.get("https://www.goibibo.com/trains/")
                driver.maximize_window()
                time.sleep(5)
                close_popup(driver)
                enter_from_location(driver, from_city)
                enter_to_location(driver, to_city)
                select_date(driver, date.day)
                click_search(driver)
                trains = fetch_train_data(driver)
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
            finally:
                driver.quit()

            if trains:
                st.success("✅ Train data fetched successfully!")
                st.json(trains)
                with open("train_data_with_prices.json", "w", encoding="utf-8") as f:
                    json.dump(trains, f, indent=4, ensure_ascii=False)
                st.download_button("📥 Download JSON", data=json.dumps(trains, indent=2), file_name="train_data.json")
            else:
                st.info("No trains found for the given route/date.")

