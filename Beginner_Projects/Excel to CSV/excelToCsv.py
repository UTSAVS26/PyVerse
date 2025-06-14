import os
import csv
import openpyxl
from pathlib import Path

def excel_to_csv(input_file, output_folder="output"):
    """
    Convert an Excel file to CSV files, one per sheet.

    Args:
        input_file (str): Path to the input Excel file (.xlsx).
        output_folder (str, optional): Directory to save output CSV files. Defaults to "output".

    Returns:
        None: Creates CSV files in the specified output folder or prints errors if they occur.
    """
    os.makedirs(output_folder, exist_ok=True)

    try:
        wb = openpyxl.load_workbook(input_file, data_only=True)
    except Exception as e:
        print(f"error opening workbook '{input_file}': {e}")
        return

    for sheet_name in wb.sheetnames:
        try:
            sheet = wb[sheet_name]
            csv_name = f"{Path(input_file).stem}_{sheet.title}.csv"
            csv_path = os.path.join(output_folder, csv_name)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in sheet.iter_rows(values_only=True):
                    writer.writerow(row)

        except Exception as e:
            print(f"error processing sheet '{sheet_name}' in '{input_file}': {e}")

if __name__ == "__main__":
    # hardcoded example - replace with your actual file and folder
    excel_to_csv("input.xlsx", "output")