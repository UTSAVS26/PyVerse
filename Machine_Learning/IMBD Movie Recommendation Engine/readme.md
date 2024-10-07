## **Movie Recommendation Engine**

###  **Goal**
The main goal of this project is to build a **Movie Recommendation Engine** that suggests movies based on the content a user prefers. The system analyzes features such as genre, plot, director, and ratings to recommend movies similar to the ones the user has watched or liked. 

### **Dataset**
- Dataset: [IMDB Top 250 Movies](https://data.world/studentoflife/imdb-top-250-lists-and-5000-or-so-data-records)
- The dataset contains 250 movies with 27 attributes including title, director, genre, plot, ratings, and more.

### **Description**
This project implements a content-based movie recommendation engine using data from IMDB's top 250 movies. By analyzing various features of the movies, such as genre, plot descriptions, and directors, the system recommends films that share similarities with the user's preferences. The recommendation is powered by the cosine similarity between movie feature vectors derived from textual data.

###  **What I had done!**
1. **Data Loading**: Loaded the dataset into a pandas DataFrame.
2. **Data Preprocessing**: 
   - Converted all plot summaries to lowercase.
   - Removed punctuation, numbers, and extra spaces.
   - Tokenized the plot summaries.
   - Removed stopwords from the tokenized plots.
3. **Feature Engineering**: Used the **TF-IDF vectorizer** to extract features from the cleaned plot summaries.
4. **Similarity Calculation**:Computed the cosine similarity matrix for all movies based on their plot summaries.
5. **Recommendation System**:Developed a function that takes a movie title as input and returns the top 10 most similar movies based on plot similarity.

###  **Models Implemented**
- **TF-IDF Vectorizer**: Extracts the important words from the movie plots.
- **Cosine Similarity**: Calculates the similarity between two movies based on their plot vectors.
I chose **TF-IDF** because it’s a powerful technique to weigh the importance of words in the context of the entire dataset. **Cosine Similarity** works well to compare text data based on vector representations.

###  **Libraries Needed**
- pandas
- numpy
- re
- nltk
- sklearn
   
###  **Exploratory Data Analysis Results**
In the exploratory data analysis (EDA), we examined the structure of the dataset, identified missing values, and explored the distribution of movie plots. The key insights include:
- **Plot Lengths**: Analyzed the distribution of movie plot lengths.
- **Word Frequency**: Visualized the most common words in the movie plots after removing stopwords.
  
###  **Performance of the Models based on the Accuracy Scores**
This is a content-based recommendation system, so accuracy is based on how well the recommended movies align with the input movie's theme or content. There are no accuracy scores per se, but the cosine similarity score shows how close the movies are in terms of plot similarity.

###  **Conclusion**
The **Movie Recommendation Engine** provides relevant movie suggestions based on plot similarity. Content-based filtering helps find movies that share thematic and narrative elements with the input movie. This can be expanded further by incorporating additional movie attributes like genre, director, or actors for more refined recommendations.

###  **Your Signature**
Shreya Tripathy  
[GitHub](https://github.com/Shreya7tripathy) | [LinkedIn](https://www.linkedin.com/in/shreyatripathy7/)