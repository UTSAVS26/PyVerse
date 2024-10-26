import streamlit as st
import pickle
from win32com.client import Dispatch


def speak(text):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(text)

tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def main():
    st.title("Email Spam Classification Application")
    st.write("Built with Streamlit & Python")
    activities = ["Classification", "About"]
    choices = st.sidebar.selectbox("Select Activities", activities)

    if choices == "Classification":
        st.subheader("Classification")
        msg = st.text_area("Enter a text", height=200)

        if st.button("Process"):
            # Preprocess and transform the input data using the same vectorizer
            data = [msg]
            data = tfidf.transform(data)
            print(data)
            # Predict the labels using the trained model
            result = model.predict(data)[0]
            print(result)
            if result == 1:
                st.success("This is a Ham Email")
                speak("This is a Ham Email")
            elif result == 0:
                st.error("This is a Spam Email")
                speak("This is a Spam Email")

if __name__ == '__main__':
    main()
