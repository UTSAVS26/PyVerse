# Excel to CSV Converter (GUI)

A simple and elegant Python desktop application that converts every sheet in an Excel (`.xlsx` or `.xls`) file into individual CSV files.

Built with **Tkinter** for the GUI and **pandas + openpyxl** for Excel processing.

---

## Requirements

- Python 3.7+
- Libraries:
  - `pandas`
  - `openpyxl`
  - `xlrd`

Install with:

```bash
pip install pandas openpyxl xlrd
````

---

## Usage

### GUI Application

1. Run the application:

   ```bash
   python excel_to_csv.py
   ```

2. Use the interface to:

   - Browse and select an Excel file.
   - Choose an output folder.
   - Click **"Convert to CSV"**.

3. One CSV will be created per sheet in the selected output folder.

---

## Function Overview

The core logic is handled by this function, also reusable in other Python projects:

```python
def excel_to_csv(input_file: str, output_folder: str) -> tuple[bool, str]
```

- **input\_file**: path to the Excel file
- **output\_folder**: target directory for CSV files

Returns a tuple:

- `True, message` on success
- `False, error_message` on failure

It:

- Converts each sheet to a `.csv`
- Sanitizes sheet names to safe filenames
- Replaces null/NaN cells with empty strings

---

## Output Structure

For an Excel file like `sales.xlsx` with sheets `Q1`, `Q2/2023`, and an unnamed sheet:

```text
output/
├── sales_Q1.csv
├── sales_Q2_2023.csv
└── sales_Sheet3.csv
```

---

## Error Handling

The application gracefully handles:

- File not found
- File permission errors (e.g. open in Excel)
- Invalid or empty sheet names
- Unexpected exceptions

Error messages are shown both in the GUI and as popups.

---

## Features

- Easy-to-use modern GUI
- Dark theme styling
- Real-time progress indicator
- Multi-threaded conversion (no freezing UI)
- Automatic folder creation if needed

---

## Project Structure

```text
Excel to CSV/
├── excel_to_csv.py              # GUI application
└── Readme.md            # Project readme
```
---