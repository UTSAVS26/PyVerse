import cv2

def main():
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)  # Change to 0 for default webcam, or provide a video file path

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Create Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and descriptors in the frame
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # Check if descriptors are found
        if descriptors is not None:
            # Draw keypoints on the frame
            frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        else:
            frame_with_keypoints = frame

        # Display the frame with keypoints
        cv2.imshow("Feature Detection", frame_with_keypoints)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
