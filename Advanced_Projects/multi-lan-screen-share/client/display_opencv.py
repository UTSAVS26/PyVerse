import cv2
import numpy as np
import io

def display_loop(frame_queue, headless=False):
    if not headless:
        cv2.namedWindow('Screen Share - OpenCV', cv2.WINDOW_NORMAL)
    while frame_queue:
        frame_data = frame_queue.pop(0)
        try:
            arr = np.frombuffer(frame_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None and not headless:
                cv2.imshow('Screen Share - OpenCV', img)
        except Exception as e:
            print(f"[OpenCV] Frame error: {e}")
        if not headless and cv2.waitKey(1) == 27:
            break
    if not headless:
        cv2.destroyAllWindows() 