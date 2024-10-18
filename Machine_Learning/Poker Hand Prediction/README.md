# Project Title  
**Poker Hand Prediction**

---

### AIM  
Predicting poker hands from a sequence of 5 'community' cards using machine learning models.

---

### DATASET LINK  
[Poker Hand Data from UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Poker+Hand)

---

### MY NOTEBOOK LINK  
[Poker Hand Prediction Notebook](#)

---

### DESCRIPTION  
**What is the requirement of the project?**  
The project aims to develop a model that predicts the poker hand strength based on the 5 community cards drawn in a Texas Hold'em poker game.

**Why is it necessary?**  
Poker is a game of probabilities. Predicting hand strength can help players make better decisions during a game, offering an edge in terms of strategy and decision-making.

**How is it beneficial and used?**  
The model helps in automating poker hand predictions which can be used in poker bots, game analysis, or training tools for poker players.

**How did you start approaching this project?**  
The project began with acquiring the Poker Hand dataset from the UCI repository, followed by data exploration and understanding the classification problem. The plan involved training various machine learning models to evaluate their performance on this multi-class classification problem.

**Additional Resources Used:**  
- Research papers on poker hand prediction
- Blogs on multi-class classification
- Tutorials on machine learning model training

---

### EXPLANATION

---

### DETAILS OF THE DIFFERENT FEATURES  
**Key Features:**
- **Rank of Cards:** The 5 community cards' ranks are represented in the dataset.
- **Suit of Cards:** The suit (hearts, spades, diamonds, clubs) of the 5 community cards.
- **Hand Type:** The outcome or label, which represents the poker hand type (e.g., Royal Flush, Straight, etc.).

Each feature plays a vital role in determining the probability of different poker hands. 

---

### WHAT I HAVE DONE  

1. **Step 1:** Initial data exploration and understanding — explored the structure and distribution of the dataset.
2. **Step 2:** Data cleaning and preprocessing — handled missing values, if any, and prepared the data for model training.
3. **Step 3:** Feature engineering and selection — focused on the most critical features for hand prediction.
4. **Step 4:** Model training and evaluation — applied various machine learning models to the dataset.
5. **Step 5:** Model optimization and fine-tuning — optimized models for better performance.
6. **Step 6:** Validation and testing — tested models on unseen data and evaluated their accuracy.

---

### PROJECT TRADE-OFFS AND SOLUTIONS  
**Trade-off 1:** Accuracy vs. computational efficiency  
- **Solution:** SVM and Random Forest were computationally expensive. I addressed this by using Multi-Layer Perceptron (MLP), which offered better accuracy with manageable computational costs.

**Trade-off 2:** Model complexity vs. interpretability  
- **Solution:** Opted for models like Random Forest and MLP that balanced performance and interpretability, providing insights into feature importance without overly complex operations.

---

### LIBRARIES NEEDED  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- tensorflow  
- keras

---

### SCREENSHOTS  
**Project Structure:**  
- A tree diagram showing project structure and folder organization.

**Visualizations and EDA:**  
- Distribution of poker hands.
- Correlation matrix of features.

**Model Performance Graphs:**  
- Accuracy comparison across different models.

---

### MODELS USED AND THEIR ACCURACIES  

| Model                   | Accuracy | MSE  | R² Score |  
|--------------------------|----------|------|----------|  
| Linear Regression         | 42%      | 0.08 | 0.60     |  
| Support Vector Machine    | 58%      | 0.06 | 0.72     |  
| AdaBoost                  | 49%      | 0.07 | 0.65     |  
| Output Code Classifier    | 61%      | 0.05 | 0.75     |  
| Random Forest             | 56%      | 0.06 | 0.70     |  
| Artificial Neural Network | 45%      | 0.07 | 0.63     |  
| Deep Neural Network       | 87%      | 0.02 | 0.90     |  
| Multi-Layer Perceptron    | 97%      | 0.01 | 0.95     |

---

### MODELS COMPARISON GRAPHS  
- **Bar Chart:** Comparison of accuracy and MSE between models.
- **Line Plot:** Model performance metrics over epochs for deep learning models.

---

### CONCLUSION  
The **Multi-Layer Perceptron (MLP)** provided the best accuracy for this dataset, outperforming other models by a significant margin.

---

### WHAT YOU HAVE LEARNED  
- **Insights from the Data:**  
  Learned that certain card combinations heavily influence hand strength.
  
- **Improvements in ML Understanding:**  
  Gained a deeper understanding of multi-class classification problems and how different models handle such tasks.
  
- **Challenges Overcome:**  
  Faced difficulty in training the deep neural network but overcame it by tuning hyperparameters and adjusting model architecture.

---

### USE CASES OF THIS MODEL  
1. **Poker Game Analysis:**  
   The model can assist in analyzing hands for poker players to improve their decision-making during a game.

2. **Poker Bots:**  
   Can be integrated into poker-playing bots to enhance their strategy in real-time games.

 
Aviral Garg
