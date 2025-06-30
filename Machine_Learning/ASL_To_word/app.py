# Import necessary libraries
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import mediapipe as mp
import threading
from tensorflow.keras.models import load_model
import pickle
import time

# Define image size for model input
IMG_SIZE = 64

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.geometry("900x700")
        self.root.configure(bg='white')

        # Load the trained model and label binarizer
        self.model = load_model("sign_model.h5")
        with open("label_binarizer.pkl", "rb") as f:
            self.lb = pickle.load(f)

        # Initialize webcam and MediaPipe hand detection
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Variables to manage collection and predictions
        self.collecting = False
        self.collected_word = ""
        self.stored_words = []
        self.last_prediction = ""
        self.last_time = time.time()

        # UI Components
        self.video_label = tk.Label(self.root)
        self.video_label.pack(pady=10)

        self.prediction_label = tk.Label(self.root, text="Prediction: None", font=("Courier", 18), bg="white")
        self.prediction_label.pack(pady=5)

        self.word_label = tk.Label(self.root, text="Word: ", font=("Courier", 18), bg="white", fg="blue")
        self.word_label.pack(pady=5)

        self.stored_label = tk.Label(self.root, text="Collected Words: ", font=("Courier", 14), bg="white", fg="green")
        self.stored_label.pack(pady=5)

        self.suggestion_frame = tk.Frame(self.root, bg="white")
        self.suggestion_frame.pack(pady=10)

        # Frame to hold control buttons
        self.button_frame = tk.Frame(self.root, bg="white")
        self.button_frame.pack(pady=10)

        # Control buttons
        self.start_button = tk.Button(self.button_frame, text="Start Collecting", command=self.start_collecting)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.end_button = tk.Button(self.button_frame, text="End & Suggest", command=self.stop_and_suggest)
        self.end_button.pack(side=tk.LEFT, padx=10)

        self.restart_button = tk.Button(self.button_frame, text="Restart", command=self.restart_session)
        self.restart_button.pack(side=tk.LEFT, padx=10)

        self.quit_button = tk.Button(self.button_frame, text="Quit", command=self.quit_app)
        self.quit_button.pack(side=tk.LEFT, padx=10)

        # Start video capture update loop
        self.update_video()

    def update_video(self):
        # Capture frame from webcam
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)  # Mirror image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        # Define left ROI (activation hand)
        left_x1, left_y1, left_x2, left_y2 = 50, 100, 250, 300
        roi_left = frame[left_y1:left_y2, left_x1:left_x2]
        rgb_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2RGB)
        results_left = self.hands.process(rgb_left)
        hand_in_left = bool(results_left.multi_hand_landmarks)

        # Define right ROI (sign prediction hand)
        right_x1, right_y1, right_x2, right_y2 = w - 300, 100, w - 100, 300
        roi_right = frame[right_y1:right_y2, right_x1:right_x2]
        rgb_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2RGB)
        results_right = self.hands.process(rgb_right)
        hand_in_right = bool(results_right.multi_hand_landmarks)

        label = "No Hand"  # Default label

        # Process prediction if hand detected in right box
        if hand_in_right:
            white_bg = np.ones((roi_right.shape[0], roi_right.shape[1], 3), dtype=np.uint8) * 255
            for hand_landmarks in results_right.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(white_bg, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            gray = cv2.cvtColor(white_bg, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            normalized = resized / 255.0
            reshaped = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            pred = self.model.predict(reshaped, verbose=0)
            label = self.lb.classes_[np.argmax(pred)]
            self.last_prediction = label

        current_time = time.time()

        # Collect letters only when both hands are present and after a time delay
        if self.collecting and hand_in_left and hand_in_right and current_time - self.last_time > 2:
            self.collected_word += self.last_prediction
            self.last_time = current_time
            self.word_label.config(text=f"Word: {self.collected_word}")

        # Draw rectangles for ROIs
        left_color = (255, 0, 0) if hand_in_left else (0, 0, 255)
        right_color = (255, 0, 0) if hand_in_right else (0, 0, 255)
        cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), left_color, 2)
        cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), right_color, 2)

        # Show predicted label
        cv2.putText(frame, f"Prediction: {label}", (right_x1, right_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.prediction_label.config(text=f"Prediction: {label}")

        # Display frame in GUI
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        # Schedule the next frame update
        self.root.after(10, self.update_video)

    def start_collecting(self):
        # Start collecting letters for forming a word
        self.collected_word = ""
        self.collecting = True
        self.word_label.config(text="Word: ")
        print("[INFO] Started collecting letters...")

    def stop_and_suggest(self):
        # Stop collecting and show suggestions
        self.collecting = False
        print(f"[INFO] Collection stopped. Word formed: {self.collected_word}")
        if self.collected_word:
            self.suggest_words(self.collected_word)

    def suggest_words(self, raw_word):
        # Clear previous suggestions
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()

        # Generate suggestion list
        suggestions = self.generate_fake_suggestions(raw_word)

        # Display suggestions as buttons
        tk.Label(self.suggestion_frame, text="Choose the correct word:", font=("Courier", 14), bg="white").pack()
        for word in suggestions:
            btn = tk.Button(self.suggestion_frame, text=word, font=("Courier", 14),
                            command=lambda w=word: self.confirm_word(w))
            btn.pack(pady=2)

    def generate_fake_suggestions(self, word):
        # Simulate suggestions (can be replaced by a spellchecker)
        base = word.upper()
        return [base, base + "E", base[:-1], base + "S"][:4]

    def confirm_word(self, word):
        # Confirm the final word and store it
        self.stored_words.append(word)
        self.stored_label.config(text=f"Collected Words: {' | '.join(self.stored_words)}")
        self.word_label.config(text="Word: ")
        self.collected_word = ""

        # Clear suggestion buttons
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()

    def quit_app(self):
        # Release camera and close app
        self.cap.release()
        self.root.destroy()

    def restart_session(self):
        # Reset the session
        self.collecting = False
        self.collected_word = ""
        self.last_prediction = ""
        self.last_time = time.time()
        self.word_label.config(text="Word: ")
        self.prediction_label.config(text="Prediction: None")

        # Clear suggestions
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()

        print("[INFO] Session restarted.")

# Run the application
if __name__ == "__main__":
    from PIL import Image, ImageTk  # PIL needed to display OpenCV frames in Tkinter
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
