# Movie Data Analysis Application

## Goal
To create an interactive application that analyzes movie data, including reviews and meta scores, and presents the information visually through various types of graphs.

## Description
This project implements a movie data analysis application using Streamlit, allowing users to explore movie ratings and trends through visualizations. Users can select from different graph types to visualize the movie dataset, facilitating a better understanding of the underlying data trends.

## What I Have Done
1. Developed an interactive user interface using Streamlit for visualizing movie data.
2. Implemented functions to create various visualizations: bar charts, pie charts, and histograms.
3. Enabled dynamic data loading from an Excel file containing movie ratings and metadata.
4. Organized code into functions for better clarity and maintainability.
5. Incorporated basic error handling for data input and visualization selection.

## Models Implemented
This project primarily focuses on:
- **Data Visualization** using Matplotlib and Seaborn libraries.
- **User Interaction** through a simple command-line interface powered by Streamlit.
- **Data Processing** using Pandas for efficient manipulation of movie datasets.

## Libraries Needed
- `streamlit`
- `pandas`
- `matplotlib`
- `seaborn`
- `openpyxl` (if using Excel files)

## Usage
Run the Streamlit application using the following command:

```bash
streamlit run your_script_name.py
# or 
python -m streamlit run yourprojectname.py

Make sure to change the directory of the Excel file before you run it.

## How to Use
- Start the application using the command above.
- Upon loading the application, users will see a title and a description of the project.
- Load your movie dataset from an Excel file.
- Choose a visualization type (Bar Chart, Pie Chart, Histogram) from the dropdown menu.
- Click the "Submit" button to generate the selected graph based on the movie data.

## Conclusion
This Movie Data Analysis Application serves as an educational tool for understanding data visualization techniques and provides a practical introduction to using Streamlit for building interactive applications. Its design is tailored for beginners to explore data analysis concepts effectively while leveraging Python's powerful data processing libraries.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Feel free to submit issues or pull requests if you have suggestions for improvements or new features!
