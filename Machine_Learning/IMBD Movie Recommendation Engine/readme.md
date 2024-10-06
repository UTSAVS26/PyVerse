# Movie Recommendation Engine

## Overview
This project builds a **Movie Recommendation Engine** using a dataset of the top 250 movies from IMDB. The recommendation system employs **Content-Based Filtering** to suggest movies based on various attributes such as Title, Director, Genre, Plot, and Ratings. By analyzing the features of the movies a user has liked, the engine provides recommendations for similar movies.

## Dataset
The dataset consists of the top 250 movies from IMDB with 27 attributes, including:
- Title
- Director
- Genre
- Plot
- Ratings
- Year of Release
- Runtime
- Country
- Language
- Cast

The dataset can be downloaded from [IMDB Top 250 Dataset](https://data.world/studentoflife/imdb-top-250-lists-and-5000-or-so-data-records).

## Libraries Used
The following Python libraries are required for this project:
1. **pandas**: For data manipulation and analysis.
2. **nltk**: For text preprocessing and natural language processing tasks.
3. **sklearn**: For machine learning algorithms and vectorization.
4. **re**: For regular expressions and text cleaning.

## Project Workflow
1. **Data Loading**: Load the IMDB dataset into a pandas DataFrame for easy manipulation and analysis.
2. **Data Preprocessing**:
   - Clean the dataset by handling missing values, removing unnecessary columns, and normalizing text data.
   - Preprocess textual attributes such as Genre, Plot, and Director using **NLTK** for tokenization, lemmatization, and removal of stopwords.
3. **Feature Engineering**: Combine relevant features like Genre, Plot, and Director to create a comprehensive profile of each movie.
4. **Vectorization**:
   - Use **TF-IDF Vectorizer** from sklearn to convert the textual data into numerical feature vectors.
5. **Similarity Calculation**:
   - Calculate the **cosine similarity** between the feature vectors to measure the similarity between movies.
6. **Movie Recommendation**:
   - Based on a user's selected movie, recommend movies with the highest cosine similarity score using Content-Based Filtering.

## Algorithms
The project leverages **Content-Based Filtering** for movie recommendations. This approach focuses on recommending movies similar to the ones the user has watched or liked, by analyzing the features of movies (e.g., genre, plot description, director, etc.).

## Insights
- **Genres Matter**: The genre of a movie plays a crucial role in determining similarity. Movies within the same genre often share thematic elements, making genre-based filtering a significant aspect of content-based recommendation systems.
  
- **Plot Descriptions Provide Context**: By utilizing the plot descriptions of movies, the engine can capture nuanced similarities between films. Movies with similar themes, narratives, or settings will be more likely to be recommended together, enriching the recommendation quality.
  
- **Director Influence**: Directors have distinctive styles that can influence the mood and tone of a movie. Including directors as a feature enhances the personalization aspect of the recommendations.

- **Cosine Similarity for Precision**: The use of cosine similarity ensures that the recommendations are not just random but mathematically precise. This method measures the angle between feature vectors, allowing the system to provide accurate suggestions based on content similarity.

- **Content Over Popularity**: Unlike collaborative filtering, which relies on user behavior and popularity, content-based filtering makes recommendations purely on movie features. This ensures that lesser-known, high-quality movies can be suggested based on a user's taste rather than just popular choices.


## Conclusion
This **Movie Recommendation Engine** provides users with personalized movie recommendations based on the attributes of their preferred movies. The system uses natural language processing and machine learning techniques to analyze movie features and suggest relevant films.