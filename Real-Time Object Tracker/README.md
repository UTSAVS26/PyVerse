# YOLO Object Detection with SORT Tracking

This repository contains a Python script for real-time object detection using YOLO (You Only Look Once) and object tracking using SORT (Simple Online and Realtime Tracking). The script processes a video file to detect objects and assign unique IDs to them for tracking.

![Captura de pantalla 2023-09-14 123838](https://github.com/Lindapazw/tracker-yolov8-sort-python/assets/88910652/56e9fb70-0801-46a1-9c08-709969a6f3ac)


## Dependencies

- `numpy`: For numerical operations.
- `cv2` (OpenCV): For image processing and computer vision tasks.
- `ultralytics`: For utilizing the YOLO model.
- `short`: For SORT tracking algorithm.

## Usage

1. Make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install numpy opencv-python ultralytics sort

```

Download the YOLO model file yolov8n.pt and place it in the same directory as the script.

Run the script using:

```bash
python main.py
```

## Explanation
Import necessary libraries:

```bash
import numpy as np
import cv2
from ultralytics import YOLO
from short import Sort
```

- Set up the video capture:
```bash
cap = cv2.VideoCapture("people.avi")
```

- Load the YOLO model:
```bash
model = YOLO("yolov8n.pt")
```

- Initialize the SORT tracker:
```bash
tracker = Sort()
```

- Process each frame in the video:
```bash
while cap.isOpened():
    status, frame = cap.read()
    if not status:
        break
```

- Perform object detection using YOLO:
```bash
results = model(frame, stream=True)
```

- Filter out detections with confidence less than 0.7 and get the bounding boxes:
```bash
for res in results:
    filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.7)[0]
    boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
```

- Update the SORT tracker with the detected bounding boxes:
```bash
tracks = tracker.update(boxes)
tracks = tracks.astype(int)
```

- Draw bounding boxes and IDs on the frame:
```bash
for xmin, ymin, xmax, ymax, track_id in tracks:
    cv2.putText(img=frame, text=f"Id: {track_id}", org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
    cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)
```

- Display the frame with detections and tracking:
```bash
Display the frame with detections and tracking:
```

- Check for the 'q' key press to exit the loop:
```bash
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

- Release the video capture and destroy any open windows:
```bash
cap.release()
cv2.destroyAllWindows()
```

## Notes
- The script uses YOLO for object detection and SORT for tracking. Make sure to have the model file (yolov8n.pt) in the same directory.
- You can replace the video file (people.avi) with your own video file for processing.
