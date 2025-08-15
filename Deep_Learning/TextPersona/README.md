# 🧠 TextPersona: Personality Type Predictor from Text Prompts

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-22%20passed-brightgreen.svg)](tests/)
[![AI](https://img.shields.io/badge/AI-Zero--shot%20Classification-orange.svg)](https://huggingface.co/)

A complete **NLP-based personality assessment tool** that predicts MBTI (Myers-Briggs Type Indicator) personality types from introspective text responses using advanced AI classification and interactive visualizations.

## 🌟 Key Features

- **🤖 AI-Powered Classification**: Zero-shot learning with pre-trained models for accurate personality prediction
- **📊 Interactive Visualizations**: Radar charts, bar graphs, and confidence gauges to understand your profile
- **💼 Career Guidance**: Suggested careers based on your personality type with detailed descriptions
- **📝 Detailed Analysis**: Comprehensive breakdown of responses with sentiment analysis and linguistic patterns
- **💾 Export Results**: Save your results for future reference in multiple formats
- **🔒 Privacy-Focused**: No personal data is stored or shared - all processing is local
- **🌐 Web Interface**: Beautiful Streamlit web application with modern UI
- **💻 CLI Mode**: Command-line interface for quick assessments and automation
- **🧪 Comprehensive Testing**: 22 test cases covering all components with 100% pass rate

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (recommended: Python 3.9+)
- **4GB+ RAM** (for AI model loading)
- **Internet connection** (for initial model download)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/textpersona.git
   cd textpersona
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:

   **🌐 Web Interface (Recommended)**:
   ```bash
   python main.py --web
   ```
   Then open your browser to `http://localhost:8501`

   **💻 Command Line Interface**:
   ```bash
   python main.py
   ```

   **🧪 Run Tests**:
   ```bash
   python main.py --test
   ```

   **📊 Demo Mode**:
   ```bash
   python demo.py
   ```

## 📁 Project Structure

```
textpersona/
│
├── prompts/
│   ├── questions.json            # 10 introspective questions
│   └── mbti_descriptions.md     # Complete MBTI type descriptions
│
├── core/
│   ├── prompt_interface.py      # CLI & web Q&A interface
│   ├── classifier_zero_shot.py  # AI-powered classification
│   ├── classifier_rules.py      # Rule-based fallback system
│   └── result_formatter.py      # Visualization & formatting
│
├── ui/
│   └── streamlit_app.py         # Beautiful web interface
│
├── data/
│   └── user_logs.json           # Optional: save anonymous user answers
│
├── tests/
│   └── test_classifier.py       # Comprehensive test suite
│
├── main.py                      # Main application entry point
├── demo.py                      # Demo script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## 🎯 How It Works

### 1. **Questionnaire System**
The system presents 10 carefully crafted introspective questions covering all MBTI dimensions:

- **I/E**: Introversion vs Extraversion
- **S/N**: Sensing vs Intuition  
- **T/F**: Thinking vs Feeling
- **J/P**: Judging vs Perceiving

**Sample Questions**:
- "What drives your decisions more: logic or emotion?"
- "Do you prefer routine and structure or spontaneity and flexibility?"
- "How do you prefer to spend your free time?"
- "When solving problems, do you focus more on concrete details or abstract concepts?"

### 2. **AI Analysis Pipeline**
Our system uses advanced NLP techniques for personality prediction:

#### **Zero-shot Classification** (Primary Method)
- Uses `facebook/bart-large-mnli` model for accurate predictions
- Analyzes text responses against MBTI type descriptions
- Provides confidence scores for each prediction
- Handles complex linguistic patterns and context

#### **Rule-based Fallback** (Backup Method)
- Keyword analysis and sentiment detection
- Linguistic pattern recognition
- Dimension scoring based on response characteristics
- Ensures reliability when AI models are unavailable

#### **Sentiment Analysis**
- VADER lexicon for emotional content analysis
- Text complexity metrics
- Language pattern recognition
- Response depth assessment

### 3. **Personality Prediction**
The system predicts one of 16 MBTI types with detailed insights:

| Type | Name | Description |
|------|------|-------------|
| INTJ | The Architect | Imaginative and strategic thinkers |
| INTP | The Logician | Innovative inventors with thirst for knowledge |
| INFJ | The Advocate | Quiet and mystical idealists |
| INFP | The Mediator | Poetic, kind and altruistic people |
| ENFJ | The Protagonist | Charismatic and inspiring leaders |
| ENFP | The Campaigner | Enthusiastic, creative free spirits |
| ENTJ | The Commander | Bold, imaginative and strong-willed leaders |
| ENTP | The Debater | Smart and curious thinkers |
| ISTJ | The Logistician | Practical and fact-minded individuals |
| ISFJ | The Defender | Dedicated and warm protectors |
| ESTJ | The Executive | Excellent administrators |
| ESFJ | The Consul | Extraordinarily caring and social |
| ISTP | The Virtuoso | Bold and practical experimenters |
| ISFP | The Adventurer | Flexible and charming artists |
| ESTP | The Entrepreneur | Smart, energetic and perceptive |
| ESFP | The Entertainer | Spontaneous, energetic entertainers |

### 4. **Detailed Insights & Visualizations**

#### **Personality Profile**
- **MBTI Type**: Your predicted personality type
- **Confidence Score**: How certain the prediction is
- **Strengths**: Key personality characteristics
- **Career Suggestions**: Recommended career paths
- **Description**: Detailed personality explanation

#### **Interactive Charts**
- **Radar Chart**: Visual representation of MBTI dimensions
- **Bar Charts**: Dimension comparison analysis
- **Confidence Gauge**: Classification certainty indicator
- **MBTI Distribution**: Top predicted types with scores

#### **Text Analysis**
- **Sentiment Analysis**: Emotional content breakdown
- **Linguistic Patterns**: Language complexity metrics
- **Response Depth**: Analysis of answer quality
- **Dimension Scores**: Detailed breakdown by MBTI axis

## 🛠️ Technology Stack

### **Core Dependencies**
- **Python 3.8+**: Modern Python with type hints and async support
- **Transformers 4.35+**: HuggingFace models for zero-shot classification
- **Torch 2.0+**: PyTorch backend for AI model inference
- **NLTK 3.8+**: Natural language processing and sentiment analysis
- **Streamlit 1.25+**: Beautiful web interface framework
- **Plotly 5.15+**: Interactive visualizations and charts
- **Pandas 1.5+**: Data manipulation and analysis
- **Scikit-learn 1.2+**: Machine learning utilities

### **AI Models**
- **Zero-shot Classification**: `facebook/bart-large-mnli` for personality prediction
- **Sentiment Analysis**: VADER lexicon for emotional content analysis
- **Rule-based Fallback**: Custom keyword matching and linguistic analysis
- **Text Processing**: NLTK tokenization and language analysis

### **Development Tools**
- **Pytest 7.0+**: Comprehensive testing framework
- **Type Hints**: Full type annotation for code quality
- **Error Handling**: Robust exception management
- **Logging**: Detailed application logging

## 📊 Example Output

### **Sample Results**
```
🧠 Your MBTI Type: ISTJ (The Logistician)

Confidence: 45.8% (via rule-based classification)

Description: Practical and fact-minded individuals, whose reliability cannot be doubted.

Strengths: Practical, Fact-minded, Reliable, Direct

Possible Careers: Accountant, Military Officer, Manager, Auditor

Analysis Method: Rule-Based
```

### **Detailed Analysis**
```
Text Analysis:
- Characters: 1,016
- Words: 159
- Sentences: 11

Sentiment Analysis:
- Positive: 0.169
- Negative: 0.047
- Neutral: 0.783
- Compound: 0.970

Dimension Analysis:
- Introversion Extraversion: I (75.0%)
- Sensing Intuition: S (80.0%)
- Thinking Feeling: T (61.5%)
- Judging Perceiving: J (75.0%)
```

## 🧪 Testing & Quality Assurance

### **Test Coverage**
- ✅ **22 comprehensive test cases** covering all components
- ✅ **100% test pass rate** with robust error handling
- ✅ **Integration tests** for end-to-end workflow
- ✅ **Unit tests** for individual components
- ✅ **Performance tests** for AI model loading

### **Test Categories**
1. **Prompt Interface Tests**: Question loading, response formatting, data saving
2. **Rule-based Classifier Tests**: Keyword analysis, dimension scoring, MBTI construction
3. **Zero-shot Classifier Tests**: AI model integration, fallback mechanisms
4. **Result Formatter Tests**: Visualization generation, export functionality
5. **Integration Tests**: End-to-end workflow testing

### **Running Tests**
```bash
# Run all tests
python tests/test_classifier.py

# Run specific test categories
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=core --cov-report=html

# Run main test suite
python main.py --test
```

## 🔧 Configuration & Customization

### **Model Selection**
Choose between different classification methods:

```python
# Zero-shot (Recommended) - Highest accuracy
classifier = ZeroShotClassifier()

# Rule-based - Offline use
classifier = RuleBasedClassifier()

# Auto-select - Best available
classifier = ZeroShotClassifier()  # Falls back to rule-based
```

### **Confidence Thresholds**
Set minimum confidence levels for predictions:

- **High (0.8+)**: Very certain predictions
- **Medium (0.6-0.8)**: Balanced accuracy and coverage
- **Low (0.4-0.6)**: More predictions but lower certainty

### **Custom Questions**
Add your own questions by editing `prompts/questions.json`:

```json
{
  "id": 11,
  "question": "Your custom question here?",
  "category": "thinking_feeling",
  "options": ["Option 1", "Option 2", "Option 3"]
}
```

### **Personality Descriptions**
Extend personality descriptions in `prompts/mbti_descriptions.md` or modify the classifier files.

## 🌐 Web Interface

### **Features**
- **Modern UI**: Beautiful Streamlit interface with responsive design
- **Interactive Forms**: Easy-to-use question interface
- **Real-time Results**: Instant personality analysis
- **Visual Charts**: Interactive radar charts and bar graphs
- **Export Options**: Download results in multiple formats
- **Settings Panel**: Configurable classification parameters

### **Usage**
1. Start the web interface: `python main.py --web`
2. Open browser to `http://localhost:8501`
3. Answer the personality questions
4. View detailed results and visualizations
5. Export your results

## 💻 Command Line Interface

### **Features**
- **Quick Assessment**: Fast personality evaluation
- **Batch Processing**: Handle multiple assessments
- **Export Results**: Save to text files
- **Verbose Output**: Detailed analysis information

### **Usage**
```bash
# Interactive CLI mode
python main.py

# Test mode
python main.py --test

# Web interface
python main.py --web

# Help
python main.py --help
```

## 🔒 Privacy & Security

### **Privacy Features**
- **Local Processing**: All analysis happens on your device
- **No Data Storage**: Responses are not saved unless explicitly requested
- **Anonymous Logs**: Optional logging without personal identifiers
- **Open Source**: Transparent code for security verification

### **Data Handling**
- **No External APIs**: All processing is local
- **Optional Logging**: User responses can be saved anonymously
- **Export Control**: Users control what data is exported
- **No Tracking**: No analytics or tracking mechanisms

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### **Development Setup**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `python tests/test_classifier.py`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### **Areas for Contribution**
- **New Questions**: Add more introspective prompts
- **Better Models**: Improve classification accuracy
- **Visualizations**: Create new chart types
- **Documentation**: Improve guides and examples
- **Testing**: Add more comprehensive tests
- **Performance**: Optimize model loading and inference
- **UI/UX**: Enhance the web interface

### **Code Standards**
- **Type Hints**: All functions should have type annotations
- **Docstrings**: Comprehensive documentation for all classes and methods
- **Tests**: New features must include tests
- **Error Handling**: Robust exception management
- **Logging**: Appropriate logging for debugging

## 📚 MBTI Framework

### **Understanding MBTI**
The MBTI (Myers-Briggs Type Indicator) framework categorizes personalities into 16 types based on four dimensions:

#### **Introversion (I) vs Extraversion (E)**
- **Introversion**: Focus on internal world, prefer solitude, think before acting
- **Extraversion**: Focus on external world, prefer social interaction, act before thinking

#### **Sensing (S) vs Intuition (N)**
- **Sensing**: Focus on concrete facts, prefer practical details, trust experience
- **Intuition**: Focus on abstract concepts, prefer possibilities, trust imagination

#### **Thinking (T) vs Feeling (F)**
- **Thinking**: Make decisions based on logic, value fairness, focus on tasks
- **Feeling**: Make decisions based on values, value harmony, focus on people

#### **Judging (J) vs Perceiving (P)**
- **Judging**: Prefer structure and planning, like closure, organized approach
- **Perceiving**: Prefer flexibility and spontaneity, like options, adaptable approach

### **Type Descriptions**
Each of the 16 MBTI types has unique characteristics:

- **INTJ - The Architect**: Strategic, independent, analytical thinkers
- **INTP - The Logician**: Innovative inventors with unquenchable thirst for knowledge
- **INFJ - The Advocate**: Quiet and mystical, yet inspiring idealists
- **INFP - The Mediator**: Poetic, kind and altruistic people
- **ENFJ - The Protagonist**: Charismatic and inspiring leaders
- **ENFP - The Campaigner**: Enthusiastic, creative and sociable free spirits
- **ENTJ - The Commander**: Bold, imaginative and strong-willed leaders
- **ENTP - The Debater**: Smart and curious thinkers
- **ISTJ - The Logistician**: Practical and fact-minded individuals
- **ISFJ - The Defender**: Dedicated and warm protectors
- **ESTJ - The Executive**: Excellent administrators
- **ESFJ - The Consul**: Extraordinarily caring and social people
- **ISTP - The Virtuoso**: Bold and practical experimenters
- **ISFP - The Adventurer**: Flexible and charming artists
- **ESTP - The Entrepreneur**: Smart, energetic and perceptive people
- **ESFP - The Entertainer**: Spontaneous, energetic entertainers

## 🚀 Roadmap

### **Planned Features**
- [ ] **Big 5 Traits**: Add Big 5 personality assessment alongside MBTI
- [ ] **Multi-language**: Support for different languages
- [ ] **API Service**: REST API for integration with other applications
- [ ] **Mobile App**: Native mobile application
- [ ] **Advanced Analytics**: More detailed personality insights
- [ ] **Comparison Tools**: Compare with famous personalities
- [ ] **Gamification**: Personality quizzes and challenges
- [ ] **Machine Learning**: Train custom models on personality data

### **Performance Improvements**
- [ ] **Faster Models**: Optimized inference speed
- [ ] **Better Accuracy**: Enhanced classification algorithms
- [ ] **Caching**: Improved response times
- [ ] **Scalability**: Handle multiple concurrent users
- [ ] **Memory Optimization**: Reduced memory footprint

### **User Experience**
- [ ] **Dark Mode**: Theme customization
- [ ] **Accessibility**: Screen reader support
- [ ] **Mobile Responsive**: Better mobile experience
- [ ] **Offline Mode**: Full offline functionality
- [ ] **Custom Themes**: User-defined color schemes

## 📞 Support & Community

### **Getting Help**
- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the code comments and docstrings
- **Examples**: See the test files for usage examples

### **Resources**
- [MBTI Foundation](https://www.myersbriggs.org/)
- [16 Personality Types](https://www.16personalities.com/)
- [Personality Psychology](https://en.wikipedia.org/wiki/Personality_psychology)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### **Community Guidelines**
- **Be Respectful**: Treat all community members with respect
- **Help Others**: Share knowledge and help newcomers
- **Follow Standards**: Adhere to coding and documentation standards
- **Report Issues**: Help improve the project by reporting bugs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MBTI Foundation**: For the personality type framework and research
- **HuggingFace**: For pre-trained models and transformers library
- **Streamlit**: For the beautiful web interface framework
- **Plotly**: For interactive visualizations and charts
- **NLTK**: For natural language processing capabilities
- **Open Source Community**: For the amazing tools and libraries

## 📈 Performance Metrics

### **Test Results**
- **22 Tests**: All passing ✅
- **Coverage**: Comprehensive component testing
- **Performance**: Fast inference with fallback systems
- **Reliability**: Robust error handling and recovery

### **Model Performance**
- **Zero-shot Accuracy**: High accuracy with pre-trained models
- **Rule-based Fallback**: Reliable keyword-based classification
- **Processing Speed**: Fast response times
- **Memory Usage**: Optimized for various hardware configurations

---

**Made with ❤️ for the personality psychology community**

*TextPersona - Understanding yourself through AI*

---

**@SK8-infi** 