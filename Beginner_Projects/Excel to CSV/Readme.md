# Excel to CSV Converter

converts every sheet of a `.xlsx` Excel file into separate CSV files.

## requirements

- python 3.x  
- library: `openpyxl`

install:
```bash
pip install openpyxl
````

## usage

### cli usage

make sure your Excel file (e.g., `input.xlsx`) is in the same directory or provide the full path:

```bash
python excel_to_csv.py input.xlsx -o output
```

this will generate csv files in the `output/` folder (created if it doesn't exist).

### python api usage

you can also import and use the function in your own python code:

```python
from excel_to_csv import excel_to_csv

excel_to_csv("your_file.xlsx", "your_output_folder")
```

## function overview

```python
def excel_to_csv(input_file: str, output_folder: str = "output") -> list[str]
```

* **input\_file**: path to `.xlsx` excel file
* **output\_folder** *(optional)*: directory where csvs will be stored
* outputs one csv per sheet named as:
  `<input_file_stem>_<sanitized_sheet_name>.csv`

## output

for an input file like `data.xlsx` with sheets `sheet1` and `2023/stats`, the output will be:

```text
output/
├── data_Sheet1.csv
└── data_2023_Stats.csv
```
---