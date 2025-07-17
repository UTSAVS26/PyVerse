import pandas as pd
import os

def load_result_data(year: str, sem: str) -> pd.DataFrame:
    year_folder = year.lower().replace(" ", "_")
    sem_file = "sem1.csv" if "1st" in sem else "sem2.csv"
    file_path = f"data/{year_folder}/{sem_file}"
    
    if not os.path.exists(file_path):
        return pd.DataFrame()  # Return empty DataFrame if file not found
    
    return pd.read_csv(file_path)  # ✅ No index=False here
