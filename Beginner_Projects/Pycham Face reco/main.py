from PIL import Image
import math
import os
import sys
import cv2
import face_recognition
import numpy as np
import datetime
import time

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    attendance_count = 0
    max_attendance_count = 3

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            try:
                # Load the image using PIL
                pil_image = Image.open(f'faces/{image}')
                # Convert to RGB format
                pil_image = pil_image.convert('RGB')
                # Save the converted image temporarily if needed
                pil_image.save(f'faces/converted_{image}')
                
                # Now load the converted image using face_recognition
                face_image = face_recognition.load_image_file(f'faces/converted_{image}')
                face_encoding = face_recognition.face_encodings(face_image)

                if face_encoding:  # Check if face encoding was found
                    self.known_face_encodings.append(face_encoding[0])
                    self.known_face_names.append(image)
                else:
                    print(f"No face found in image: {image}")

            except Exception as e:
                print(f"Error processing {image}: {e}")

        print("Known faces:", self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found!')

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                if 'unknown' not in name:
                    self.attendance_count += 1
                    print(f'Attendance {self.attendance_count} recorded at {datetime.datetime.now()}')

                    if self.attendance_count >= self.max_attendance_count:
                        print('Attendance completed!')
                        self.attendance_count = 0  # Reset the attendance count
                        time.sleep(10)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
