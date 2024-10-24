## **PyTorch Based AI Chatbot**




### 🎯 **Goal**

Python Project - This ChatBot allows programmers to attach it to any website where normal users can use it.

Modules Used:
    1. Pytorch
    2. NLTK
    3. NumPy
    4. Random
    5. JSON
    6. Flask
	

# Chatbot Deployment with Flask and JavaScript
How to deploy:
- Deploy within Flask app with jinja2 template

## Initial Setup:
```
$ cd chatbot
$ py -m venv venv
$ . venv/scripts/activate
```
Install dependencies
```
$ (venv) pip install Flask torch torchvision nltk
```
Install nltk package
```
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')
```
Modify `intents.json` with different intents and responses for your Chatbot as per your need

Run
```
$ (venv) python train.py
```
This will dump data.pth file. And then run
the following command to test it in the console.

```
Run the chatbot with frontend in cmd or editor
```
```
$python app.py
```

### ✒️ **Your Signature**

`Ankan Mukhopadhyay`
[GitHub Profile](https://github.com/Peart-Guy) | [LinkedIn](https://www.linkedin.com/in/ankan-mukhopadhyay-06baa4315/)

