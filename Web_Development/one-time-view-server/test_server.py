import os
import time
import tempfile
import pytest
from fastapi.testclient import TestClient
from server import app, UPLOAD_DIR

client = TestClient(app)

def test_upload_and_download():
    # Upload a file
    test_content = b"Hello, test!"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(test_content)
        tmp.flush()
        tmp.seek(0)
        response = client.post(
            "/upload",
            files={"file": ("test.txt", tmp, "text/plain")},
            data={"expiry": 60}
        )
    assert response.status_code == 200
    assert b"file_url" in response.content or b"/file/" in response.content
    # Extract token from response
    import re
    match = re.search(rb'/file/([\w\-]+)', response.content)
    assert match, "No file URL found in response"
    token = match.group(1).decode()
    # Download the file
    file_url = f"/file/{token}"
    resp = client.get(file_url)
    assert resp.status_code == 200
    assert resp.content == test_content
    # Second download should fail (one-time)
    resp2 = client.get(file_url)
    assert resp2.status_code == 410

def test_expiry():
    # Upload a file with short expiry
    test_content = b"Expire soon!"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(test_content)
        tmp.flush()
        tmp.seek(0)
        response = client.post(
            "/upload",
            files={"file": ("expire.txt", tmp, "text/plain")},
            data={"expiry": 1}
        )
    assert response.status_code == 200
    import re
    match = re.search(rb'/file/([\w\-]+)', response.content)
    assert match, "No file URL found in response"
    token = match.group(1).decode()
    file_url = f"/file/{token}"
    # Print expiry info
    from server import file_store
    print(f"TEST: Now={int(time.time())}, Expiry={file_store[token]['expiry']}")
    # Wait for expiry
    time.sleep(2)
    print(f"TEST: After sleep, Now={int(time.time())}, Expiry={file_store[token]['expiry']}")
    resp = client.get(file_url)
    assert resp.status_code == 410

def test_qr_code():
    # Upload a file
    test_content = b"QR test!"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(test_content)
        tmp.flush()
        tmp.seek(0)
        response = client.post(
            "/upload",
            files={"file": ("qr.txt", tmp, "text/plain")},
            data={"expiry": 60}
        )
    assert response.status_code == 200
    import re
    match = re.search(rb'/file/([\w\-]+)', response.content)
    assert match, "No file URL found in response"
    token = match.group(1).decode()
    # Get QR code
    qr_url = f"/file/qr/{token}"
    resp = client.get(qr_url)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

def teardown_module(module):
    # Clean up uploads directory after tests
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        try:
            os.remove(fpath)
        except Exception:
            pass 