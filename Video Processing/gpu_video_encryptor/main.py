import argparse
import hashlib
import numpy as np

from encryptor import encrypt_frame, decrypt_frame
from gpu_acceleration import gpu_encrypt_decrypt_frame  # Unified GPU function
from video_processor import extract_frames, save_video
from utils import derive_key


def main():
    parser = argparse.ArgumentParser(description="GPU Video Encryptor/Decryptor")
    parser.add_argument("mode", choices=["encrypt", "decrypt"], help="Mode: encrypt or decrypt")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("password", help="Password for encryption/decryption")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    args = parser.parse_args()

    # Derive AES key from password
    key = derive_key(args.password)

    # IV derived for demo purposes; in production, securely generate and store it
    iv = hashlib.sha256(args.password.encode() + b"iv_salt").digest()[:16]

    # Extract frames from input video
    frames = extract_frames(args.video_path)
    processed_frames = []

    for frame in frames:
        frame_bytes = frame.tobytes()

        if args.gpu:
            # Unified GPU AES-CBC encryption/decryption
            enc_bytes = gpu_encrypt_decrypt_frame(np.frombuffer(frame_bytes, dtype=np.uint8), key, iv, mode=args.mode)
        else:
            if args.mode == "encrypt":
                enc_bytes = encrypt_frame(frame_bytes, key, iv)
            else:
                enc_bytes = decrypt_frame(frame_bytes, key, iv)

        processed_frame = np.frombuffer(enc_bytes, dtype=np.uint8).reshape(frame.shape)
        processed_frames.append(processed_frame)

    out_path = f"output_{args.mode}.avi"
    save_video(processed_frames, out_path)
    print(f"{args.mode.title()}ed video saved to: {out_path}")


if __name__ == "__main__":
    main()
