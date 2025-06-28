
import cv2

def extract_frames(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, path, fps=30):
    if not frames:
        raise ValueError("Frames list cannot be empty")
    
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    if not out.isOpened():
        raise ValueError(f"Unable to create video file: {path}")
    
    for f in frames:
        out.write(f)
    out.release()
