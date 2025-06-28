
import argparse
from encryptor import encrypt_frame, decrypt_frame
from gpu_acceleration import gpu_encrypt_frame
from video_processor import extract_frames, save_video
from utils import derive_key, generate_iv
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="GPU Video Encryptor")
    parser.add_argument("mode", choices=["encrypt", "decrypt"])
    parser.add_argument("video_path")
    parser.add_argument("password")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

# At the top of Video Processing/gpu_video_encryptor/main.py
import argparse
import hashlib
from encryptor import encrypt_frame, decrypt_frame

# …

key = derive_key(args.password)
# For demo purposes – in production, store IV with encrypted data
iv = hashlib.sha256(args.password.encode() + b"iv_salt").digest()[:16]
frames = extract_frames(args.video_path)
    processed_frames = []

    for frame in frames:
        frame_bytes = frame.tobytes()
        if args.gpu:
            enc_bytes = gpu_encrypt_frame(np.frombuffer(frame_bytes, dtype=np.uint8), key)
        else:
            enc_bytes = encrypt_frame(frame_bytes, key, iv) if args.mode == "encrypt" else decrypt_frame(frame_bytes, key, iv)
        processed_frame = np.frombuffer(enc_bytes, dtype=np.uint8).reshape(frame.shape)
        processed_frames.append(processed_frame)

    out_path = f"output_{args.mode}.avi"
    save_video(processed_frames, out_path)
    print(f"{args.mode.title()}ed video saved to: {out_path}")

if __name__ == "__main__":
    main()
