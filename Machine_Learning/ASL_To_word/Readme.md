
# SignSpark

**SignSpark** is a GUI-based application that translates American Sign Language (ASL) gestures into corresponding English alphabets and forms words in real-time. It aims to bridge the communication gap between hearing and speech-impaired individuals and the rest of the world using computer vision and machine learning.

## 🧠 Features

- 📷 Real-time detection of ASL signs using webcam input  
- 🔤 Translates signs into corresponding alphabets  
- 📝 Forms words from detected signs  
- 🖥️ User-friendly graphical interface for ease of interaction  
- 💾 Option to reset or clear the formed text  
- 🔊 Potential for future voice output of recognized words  

## 🛠️ Tech Stack

- Python  
- OpenCV – for capturing and processing video frames  
- TensorFlow/Keras – for ASL gesture classification model  
- Tkinter – for building the GUI (choose based on your usage)  
- NumPy, PIL – for image preprocessing  

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.7+  
- pip
- 
### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vaish-011/SignSpark
   cd SignSpark
    ```

2. Install Dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

   ```bash
   python app.py
   ```

## 📁 Project Structure

```
signspark/
│
├── sign_language_dataset/                  #Dataset
├── app.py                 # application
├── label_binarizer.pkl     
├── model3.ipynb            # model for data collection, training and prediction
├── sign_model.h5           # trained model 
├── requirements.txt
└── README.md
```

## 📷 Demo Video

[SignSpark Demo](video/demo.mp4)

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.


## 👩‍💻 Developed By

**Muskan Tomar**
[LinkedIn](http://linkedin.com/in/muskan-tomar-1414962b6) | [GitHub](https://github.com/Vaish-011)

