# 👁️ Eye Strain Predictor

A machine learning-powered web application that predicts digital eye strain risk based on user screen usage habits and lifestyle factors.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Smart Prediction**: Uses Random Forest classifier to assess eye strain risk
- **Interactive Web Interface**: Beautiful Streamlit-based user interface
- **Comprehensive Analysis**: Considers 11 different factors affecting eye health
- **Visual Results**: Probability charts and detailed recommendations
- **Professional Grade**: Well-documented, type-hinted, and error-free code

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **Clone or download the project files**
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the training dataset:**
   ```bash
   python generate_dataset.py
   ```

4. **Train the machine learning model:**
   ```bash
   python train_model.py
   ```

5. **Run the web application:**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and go to `http://localhost:8501`

## 📁 Project Structure

```
Eye_Strain_Predictor/
│
├── app.py                 # Main Streamlit web application
├── train_model.py         # Model training and evaluation
├── generate_dataset.py    # Synthetic dataset generation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
│
├── eye_strain_dataset.csv    # Generated training data
├── eye_strain_model.joblib   # Trained ML model
├── feature_importance.png    # Feature importance plot
└── confusion_matrix.png      # Model performance visualization
```

## 🧠 How It Works

### Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Features**: 11 input variables including screen time, brightness, environment factors
- **Target**: 4 risk levels (None, Mild, Moderate, Severe)
- **Accuracy**: ~85-90% on synthetic test data

### Input Features

| Feature | Description | Range/Type |
|---------|-------------|------------|
| Age | User's age | 16-65 years |
| Screen Time | Daily screen usage | 0.5-16 hours |
| Screen Brightness | Display brightness level | 10-100% |
| Screen Distance | Distance from eyes to screen | 20-100 cm |
| Room Lighting | Ambient lighting quality | Poor/Adequate |
| Blink Rate | Blinks per minute | 5-25 blinks/min |
| Break Frequency | Screen breaks per hour | 0-6 breaks/hour |
| Sleep Quality | Sleep quality rating | 1-5 scale |
| Blue Light Filter | Use of blue light protection | Yes/No |
| Eye Exercises | Regular eye exercise practice | Yes/No |
| Previous Eye Problems | Existing eye conditions | Yes/No |

### Risk Levels & Recommendations

- **🟢 None**: Healthy screen habits, maintain current routine
- **🟡 Mild**: Minor improvements needed, implement 20-20-20 rule
- **🟠 Moderate**: Action required, reduce screen time and improve ergonomics
- **🔴 Severe**: Immediate changes needed, consider professional consultation

## 🛠️ Technical Details

### Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **joblib**: Model serialization

### Code Quality Features

- ✅ **Type Hints**: Full type annotation for better code clarity
- ✅ **Documentation**: Comprehensive docstrings for all functions
- ✅ **Error Handling**: Robust error handling and user feedback
- ✅ **Code Style**: PEP 8 compliant formatting
- ✅ **Modular Design**: Clean separation of concerns
- ✅ **Professional Structure**: Industry-standard project organization

## 📊 Model Performance

The Random Forest model achieves high accuracy through:

- **Feature Engineering**: Realistic synthetic data generation
- **Balanced Dataset**: Equal representation of all risk levels
- **Cross-Validation**: Robust model validation techniques
- **Feature Importance**: Analysis of most predictive factors

Key insights from the model:
- Screen time and break frequency are the most important predictors
- Environmental factors significantly impact eye strain risk
- Personal habits like eye exercises provide substantial protection

## 🎯 Use Cases

- **Personal Health**: Self-assessment of digital eye strain risk
- **Workplace Wellness**: Employee health screening tools
- **Educational Purpose**: Learning about eye health and ML
- **Healthcare**: Preliminary screening (not a medical diagnosis)

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It is not intended to provide medical advice or diagnosis. For serious eye health concerns, please consult with a qualified eye care professional.

## 🤝 Contributing

This project follows professional development standards:

1. **Code Quality**: All code includes type hints and comprehensive documentation
2. **Testing**: Functions are designed for easy testing and validation
3. **Structure**: Modular design allows for easy extension and modification
4. **Standards**: Follows Python best practices and PEP guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Future Enhancements

- **Real Data Integration**: Connect with actual eye tracking devices
- **Mobile App**: React Native or Flutter mobile application
- **Advanced ML**: Deep learning models for improved accuracy
- **Integration**: API endpoints for third-party applications
- **Personalization**: User profiles and historical tracking

## 📧 Support

For questions, issues, or contributions, please feel free to reach out or create an issue in the repository.

---

**Made with ❤️ for digital health and wellness**
