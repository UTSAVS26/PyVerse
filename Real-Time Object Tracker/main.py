import numpy as np
import cv2
from ultralytics import YOLO
from short import Sort

if __name__ == '__main__':
    cap = cv2.VideoCapture("people2.avi")
    model = YOLO("yolov8n.pt")
    tracker = Sort()

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        results = model(frame, stream=True)

        for res in results:
            # Extract boxes and scores without converting to int (keeps precision for IoU)
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.3)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices]
            scores = res.boxes.conf.cpu().numpy()[filtered_indices][:, None]
            dets = np.hstack((boxes, scores))

            tracks = tracker.update(dets)
            tracks = tracks.astype(int)  # For display purposes

            for xmin, ymin, xmax, ymax, track_id in tracks:
                cv2.putText(
                    img=frame, 
                    text=f"Id: {track_id}", 
                    org=(xmin, ymin - 10), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2, 
                    color=(0, 255, 0), 
                    thickness=2
                )
                cv2.rectangle(
                    img=frame, 
                    pt1=(xmin, ymin), 
                    pt2=(xmax, ymax), 
                    color=(0, 255, 0), 
                    thickness=2
                )

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
