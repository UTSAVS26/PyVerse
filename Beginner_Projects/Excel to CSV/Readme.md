# Excel to CSV Converter

Converts every sheet of a `.xlsx` Excel file into separate CSV files.

## 📦 Requirements

- Python 3.x
- Library: `openpyxl`

Install:
```bash
pip install openpyxl
````

## 🚀 Usage

### CLI Usage

Make sure your Excel file (e.g., `input.xlsx`) is in the same directory or provide the path:

```bash
python excelToCsv.py input.xlsx -o output
```

This will generate CSV files in the `output/` folder (created if it doesn't exist).

### Python API Usage

You can also import and use the function in your own Python code:

```python
from excelToCsv import excel_to_csv

excel_to_csv("your_file.xlsx", "your_output_folder")
```

## 🧠 Function Overview

```python
def excel_to_csv(input_file: str, output_folder: str = "output") -> None
```

* **input\_file**: Path to `.xlsx` Excel file
* **output\_folder** *(optional)*: Directory where CSVs will be stored
* Outputs one CSV per sheet named as:
  `<input_file_stem>_<sanitized_sheet_name>.csv`

## 📁 Output

For an input file like `data.xlsx` with sheets `Sheet1` and `2023/Stats`, the output will be:

```
output/
├── data_Sheet1.csv
└── data_2023_Stats.csv
```
---