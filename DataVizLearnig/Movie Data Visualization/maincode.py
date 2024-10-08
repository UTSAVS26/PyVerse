import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of the application
st.title("Movie Data analysis")
st.write(
    "This project analyses the movie data such as ,reviews,meta score etc and displays them in the form of a graph"
)

# Load your dataset
data = pd.read_excel(
    r"C:\Users\Ananya\OneDrive\Documents\GitHub\PyVerse\DataVizLearnig\Movie Data Visualization\MovieRatings.xlsx"
)

# Create a dropdown menu for selecting options
options = ["Barchart", "PieChart", "Histogram"]
selected_option = st.selectbox("Choose an option", options)


# Function to create and display a bar chart
def show_barchart():
    ratings = data["IMDB_Ratings"].value_counts()
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("Blues")
    colors = [cmap(i / len(ratings)) for i in range(5, len(ratings))]

    plt.bar(ratings.index.astype(str), ratings.values, width=0.5, color=colors)
    plt.xlabel("IMDB Ratings")
    plt.ylabel("Frequency")
    plt.title("Bar Chart for Ratings vs Range")
    st.pyplot(plt)


# Function to create and display a pie chart
def show_piechart():
    data["Genre"] = data["Genre"].str.split(", ")  # Split genres if multiple
    exploded_df = data.explode("Genre")
    genre_counts = exploded_df["Genre"].value_counts()

    # Create a gradient color from dark pink to light pink
    colors = [
        (1, 0.08, 0.58, 1 - i / len(genre_counts)) for i in range(len(genre_counts))
    ]

    plt.figure(figsize=(8, 6))
    plt.pie(
        genre_counts,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        wedgeprops={"edgecolor": "black"},  # Black border between slices
    )
    plt.title("Genre Distribution of Top 1000 Movies")
    st.pyplot(plt)


# Function to create and display a histogram
def show_histogram():
    # Ensure 'Meta_score' is in the correct format
    values = pd.to_numeric(data["Meta_score"], errors="coerce")

    # Check for NaN values and handle them (e.g., drop or fill)
    values = values.dropna()  # Drop NaN values

    # Create a histogram for Metascore with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(values, bins=30, kde=True, color="orange", stat="density", alpha=0.5)
    plt.title("Histogram of Metascore with KDE")
    plt.xlabel("Metascore")
    plt.ylabel("Density")
    plt.grid(axis="y")

    st.pyplot(plt)


# Run the appropriate function based on user selection
if st.button("Submit"):
    if selected_option == "Barchart":
        show_barchart()
    elif selected_option == "PieChart":
        show_piechart()
    elif selected_option == "Histogram":
        show_histogram()
    else:
        st.error("No valid option selected.")
