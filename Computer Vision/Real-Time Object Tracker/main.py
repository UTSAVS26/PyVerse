import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort  # Simple Online & Realtime Tracking

def run_tracker(cap, model, tracker, mode="cpu"):
    total_frames = 0
    start_time = time.time()

    window_name = f"YOLOv8 - {mode.upper()}"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True, verbose=False)

        # always call update(), even if there are no detections this frame
        dets = np.empty((0, 5))          # xmin, ymin, xmax, ymax, score
        for res in results:
            if not hasattr(res, "boxes") or res.boxes is None:
                continue

            conf = res.boxes.conf
            if conf is None or len(conf) == 0:
                continue

            mask = conf > 0.3
            if mask.sum() == 0:
                continue

            boxes = res.boxes.xyxy[mask].cpu().numpy()
            scores = conf[mask].cpu().numpy().reshape(-1, 1)
            # accumulate detections from every result
            dets = np.vstack((dets, np.hstack((boxes, scores))))

        # now update once per frame, even if dets is empty
        tracks = tracker.update(dets).astype(int)

        for xmin, ymin, xmax, ymax, track_id in tracks:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        total_frames += 1
        elapsed = time.time() - start_time
        fps_text = f"{mode.upper()} FPS: {total_frames / elapsed:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    fps = total_frames / (time.time() - start_time)
    cap.release()
    cv2.destroyWindow(window_name)
    return fps

def benchmark_model(video_path):
    print("🔵 Running on CPU...")
    cap_cpu = cv2.VideoCapture(video_path)
    model_cpu = YOLO("yolov8n.pt")
    tracker_cpu = Sort()
    fps_cpu = run_tracker(cap_cpu, model_cpu, tracker_cpu, mode="cpu")
    print(f"🔵 CPU FPS: {fps_cpu:.2f}")

    print("🟡 Running on GPU...")
    cap_gpu = cv2.VideoCapture(video_path)
    model_gpu = YOLO("yolov8n.pt").to('cuda')
    tracker_gpu = Sort()
    fps_gpu = run_tracker(cap_gpu, model_gpu, tracker_gpu, mode="gpu")
    print(f"🟡 GPU FPS: {fps_gpu:.2f}")

    print("🟢 Running on TensorRT...")
    cap_trt = cv2.VideoCapture(video_path)
    model_trt = YOLO("yolov8n.engine")  # Exported TensorRT engine
    tracker_trt = Sort()
    fps_trt = run_tracker(cap_trt, model_trt, tracker_trt, mode="tensorrt")
    print(f"🟢 TensorRT FPS: {fps_trt:.2f}")

    print("\n📊 Final Benchmark Results:")
    print(f"CPU      : {fps_cpu:.2f} FPS")
    print(f"GPU      : {fps_gpu:.2f} FPS (↑ {fps_gpu/fps_cpu:.2f}x)")
    print(f"TensorRT : {fps_trt:.2f} FPS (↑ {fps_trt/fps_cpu:.2f}x)")

if __name__ == "__main__":
    benchmark_model("traffic.avi")
