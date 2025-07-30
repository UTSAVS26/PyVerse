import threading
import time
import os

def cleanup_expired_files(file_store, upload_dir):
    now = int(time.time())
    tokens_to_delete = []
    for token, entry in list(file_store.items()):
        if entry["accessed"] or now > entry["expiry"]:
            try:
                os.remove(entry["path"])
            except Exception:
                pass
            # Remove QR code if exists
            qr_path = os.path.join(upload_dir, f"{token}_qr.png")
            if os.path.exists(qr_path):
                os.remove(qr_path)
            tokens_to_delete.append(token)
    for token in tokens_to_delete:
        file_store.pop(token, None)

def start_cleanup_scheduler(file_store, upload_dir, interval=60):
    def run():
        while True:
            cleanup_expired_files(file_store, upload_dir)
            time.sleep(interval)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
