import speech_recognition as sr
import webbrowser
import pyttsx3
import musicLibrary
import requests
import traceback  # For detailed error information

recognizer = sr.Recognizer()
engine = pyttsx3.init()
newsapi = "91267c82ecf0473a9474d0d131d5a87c"

def speak(text):
    engine.say(text)
    engine.runAndWait()

def processCommand(c):
    try:
        if "open google" in c.lower():
            webbrowser.open("https://google.com")
        elif "open youtube" in c.lower():
            webbrowser.open("https://youtube.com")
        
        # Handling the play command
        elif c.lower().startswith("play"):
            song = " ".join(c.lower().split(" ")[1:])
            if song in musicLibrary.music:
                link = musicLibrary.music[song]
                webbrowser.open(link)
            else:
                speak(f"Sorry, I couldn't find {song} in the music library.")

        elif "news" in c.lower():
            try:
                r = requests.get(f"https://newsapi.org/v2/top-headlines?country=in&apiKey={newsapi}")
                if r.status_code == 200:
                    # Parse the JSON response
                    data = r.json()

                    # Extract the articles
                    articles = data.get('articles', [])

                    # Speak the headlines
                    for article in articles:
                        speak(article['title'])
                else:
                    speak("I couldn't fetch the news at the moment.")
            except Exception as news_error:
                speak("There was an error fetching the news.")
                print(f"Error fetching news: {news_error}")
                traceback.print_exc()  # To log the detailed traceback

    except Exception as e:
        print(f"Error in processCommand: {e}")
        traceback.print_exc()  # Print the full traceback for debugging

if __name__ == "__main__":
    speak("Initializing Jarvis....")

    while True:
        try:
            print("recognizing...")
            with sr.Microphone() as source:
                print("Listening...")
                audio = recognizer.listen(source)
            
            # Recognize the wake word
            word = recognizer.recognize_google(audio)
            if word.lower() == "jarvis":
                speak("yes maam")
                
                # Listen for the actual command
                with sr.Microphone() as source:
                    print("Jarvis Active...")
                    audio = recognizer.listen(source)
                    command = recognizer.recognize_google(audio)

                    processCommand(command)

        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            speak("Sorry, I didn't catch that.")
        
        except sr.RequestError as req_err:
            print(f"Could not request results from Google Speech Recognition service; {req_err}")
            speak(f"Could not request results from Google Speech service; {req_err}")
        
        except Exception as e:
            print(f"Error; {e}")
            traceback.print_exc()  # Print the full traceback for debugging
            speak("Something went wrong, please try again.")
