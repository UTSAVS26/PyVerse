# 📅 MoodMeet: AI-Powered Meeting Mood Analyzer

## 🎯 Project Overview

**MoodMeet** analyzes meeting transcripts and team chat logs to detect emotional tone, sentiment trends, and group mood dynamics. It uses advanced NLP techniques including sentiment analysis, topic clustering, and interactive visualizations to provide insights into team productivity and emotional well-being.

## ✨ Features

- **Sentiment Analysis**: Multi-model approach using VADER, TextBlob, and Transformer-based models
- **Mood Clustering**: K-Means and LDA topic grouping to identify discussion themes
- **Keyword Extraction**: RAKE, YAKE, and tf-idf for key phrase identification
- **Interactive Visualizations**: Timeline charts, heatmaps, and sentiment distributions
- **Real-time Analysis**: Upload transcripts or paste text for immediate insights
- **Speaker Analysis**: Track individual participant mood patterns

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd MoodMeet

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run ui/streamlit_app.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📁 Project Structure

```
moodmeet/
│
├── data/
│   ├── example_transcripts/       # Sample meeting transcripts
│   └── uploader.py               # Text input handling
│
├── analysis/
│   ├── sentiment_analyzer.py     # Multi-model sentiment analysis
│   ├── mood_clustering.py        # Topic clustering algorithms
│   └── keyword_extractor.py      # Keyword extraction methods
│
├── visualization/
│   ├── mood_timeline.py          # Timeline visualizations
│   └── heatmap_generator.py      # Heatmap and distribution plots
│
├── ui/
│   └── streamlit_app.py          # Main Streamlit application
│
├── models/
│   └── transformer_sentiment.py  # Transformer-based sentiment models
│
├── tests/
│   └── test_sentiment.py         # Comprehensive test suite
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 💡 Usage Examples

### Input Format

```
Alice: We're falling behind schedule.
Bob: Let's regroup and finish the draft today.
Carol: Honestly, I'm feeling a bit burned out.
David: I think we can make it work if we focus.
```

### Output Insights

#### Sentiment Summary
- **Overall Sentiment**: Slightly Negative
- **Most Negative Statement**: "I'm feeling a bit burned out."
- **Positive Ratio**: 34%
- **Sentiment Trend**: Improving over time

#### Mood Clusters
- 🧠 **Focus & Deadlines**: Schedule discussions and planning
- 😟 **Frustration/Stress**: Burnout and pressure mentions
- 💡 **Planning Next Steps**: Solution-oriented discussions

#### Visualizations
- Timeline charts showing mood evolution
- Heatmaps of sentiment by speaker/topic
- Keyword clouds highlighting key themes

## 🧠 NLP Techniques

### Sentiment Analysis
- **VADER**: Rule-based sentiment analysis optimized for social media
- **TextBlob**: Simple and effective polarity scoring
- **Transformer Models**: Advanced neural network-based analysis

### Topic Clustering
- **K-Means**: Traditional clustering for topic groups
- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic modeling
- **UMAP + HDBSCAN**: Advanced clustering for complex patterns

### Keyword Extraction
- **RAKE**: Rapid Automatic Keyword Extraction
- **YAKE**: Yet Another Keyword Extractor
- **TF-IDF**: Term frequency-inverse document frequency

## 🔧 Configuration

The application supports various configuration options:

- **Model Selection**: Choose between different sentiment analysis models
- **Clustering Parameters**: Adjust topic clustering sensitivity
- **Visualization Themes**: Customize chart colors and styles
- **Export Options**: Save results as PDF, CSV, or interactive HTML

## 🧪 Testing

The project includes comprehensive tests covering:

- Sentiment analysis accuracy
- Clustering algorithm performance
- Keyword extraction quality
- Visualization generation
- Data processing pipelines

Run tests with:
```bash
pytest tests/ -v --cov=. --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NLTK and spaCy for NLP capabilities
- HuggingFace Transformers for advanced models
- Streamlit for the interactive web interface
- Plotly and Matplotlib for visualizations

## 📞 Support

For questions, issues, or contributions, please open an issue on GitHub or contact the development team.

---

**MoodMeet** - Understanding team dynamics through AI-powered sentiment analysis 🧠💬📊 