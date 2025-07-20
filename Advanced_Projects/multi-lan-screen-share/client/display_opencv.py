import cv2
import numpy as np
import io

def display_loop(frame_queue, headless=False):
    if frame_queue is None:
        raise ValueError("frame_queue cannot be None")
    if not hasattr(frame_queue, 'pop'):
        raise TypeError("frame_queue must support pop() method")
    # …rest of the function…
    if not headless:
        try:
            cv2.namedWindow('Screen Share - OpenCV', cv2.WINDOW_NORMAL)
        except cv2.error as e:
            print(f"[ERROR] Failed to create OpenCV window: {e}")
            return
    while frame_queue:
        try:
            frame_data = frame_queue.pop(0)
        except IndexError:
            # Queue was emptied by another thread
            break
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