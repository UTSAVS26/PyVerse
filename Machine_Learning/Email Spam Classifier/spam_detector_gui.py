import pickle
from tkinter import messagebox
import tkinter as tk
from win32com.client import Dispatch

def text_to_speech(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

model = pickle.load(open('model.pkl', 'rb'))
tfidf= pickle.load(open('tfidf.pkl', 'rb'))

def result():
    data = [text.get("1.0", "end-1c")]  # get text from 1st line to last line
    data = tfidf.transform(data)

    my_prediction = model.predict(data)
    print(my_prediction[0])
    if my_prediction[0] == 0:
        text_to_speech("This is a Spam mail")
        messagebox.showinfo("Result", "This is a Spam mail")
    elif my_prediction[0] == 1:
        text_to_speech("This is a Ham mail")
        messagebox.showinfo("Result", "This is a Ham mail")

root = tk.Tk()
root.geometry("400x400")

l2 = tk.Label(root, text="Email Spam Classification Application")
l2.pack()

l1 = tk.Label(root, text="Enter Your Message:")
l1.pack()

text = tk.Text(root, width=40, height=10)
text.pack()

button = tk.Button(root, text="Click", command=result, width=20, height=2,
                   font=("Arial", 14, "bold"), fg="white",bg="darkblue",
                   activebackground="darkblue", activeforeground="white",
                   relief=tk.RAISED)
button.pack(pady=10)

root.mainloop()
