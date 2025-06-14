# Excel to CSV Converter

Converts Excel (.xlsx) sheets to separate CSV files.

## Requirements
- Python 3.x
- Libraries: `openpyxl`, `pathlib`, `csv` (built-in)

Install:
```bash
pip install openpyxl
```

## Usage
Run:
```bash
python excel_to_csv.py
```

Edit script to set your Excel file and output folder:
```python
excel_to_csv("your_file.xlsx", "your_output_folder")
```

## Code
- `excel_to_csv(input_file, output_folder="output")`: Converts each Excel sheet to a CSV file in the specified folder.
- Handles errors for invalid files/sheets.
- Output: One CSV per sheet, named `<input_file_stem>_<sheet_name>.csv`.
