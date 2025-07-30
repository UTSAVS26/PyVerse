import os
import shutil
import time
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from utils.token_generator import generate_token
from utils.cleanup_scheduler import start_cleanup_scheduler
import qrcode

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory store: token -> {filename, path, expiry, accessed}
file_store = {}
# Track tokens that have been accessed or expired
used_tokens = set()

# Configurable expiry in seconds
DEFAULT_EXPIRY = 600  # 10 minutes

@app.on_event("startup")
def startup_event():
    start_cleanup_scheduler(file_store, UPLOAD_DIR)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), expiry: int = Form(DEFAULT_EXPIRY)):
    token = generate_token()
    filename = f"{token}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    expiry_time = int(time.time()) + expiry
    file_store[token] = {
        "filename": file.filename,
        "path": file_path,
        "expiry": expiry_time,
        "accessed": False
    }
    url = request.url_for("get_file", token=token)
    # Generate QR code
    qr = qrcode.make(str(url))
    qr_path = os.path.join(UPLOAD_DIR, f"{token}_qr.png")
    qr.save(qr_path)
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "file_url": url,
        "qr_code": f"/file/qr/{token}"
    })

@app.get("/file/{token}")
async def get_file(token: str, background_tasks: BackgroundTasks):
    entry = file_store.get(token)
    if not entry:
        if token in used_tokens:
            return HTMLResponse("File already accessed or expired.", status_code=410)
        return HTMLResponse("File not found or expired.", status_code=404)
    if int(time.time()) > entry["expiry"]:
        print(f"DEBUG: Now={int(time.time())}, Expiry={entry['expiry']}, Token={token}")
        # Cleanup
        try:
            os.remove(entry["path"])
        except Exception:
            pass
        entry["accessed"] = True
        used_tokens.add(token)
        qr_path = os.path.join(UPLOAD_DIR, f"{token}_qr.png")
        if os.path.exists(qr_path):
            os.remove(qr_path)
        return HTMLResponse("File expired.", status_code=410)
    if entry["accessed"]:
        used_tokens.add(token)
        return HTMLResponse("File already accessed.", status_code=410)
    entry["accessed"] = True
    response = FileResponse(entry["path"], filename=entry["filename"])
    def cleanup():
        try:
            os.remove(entry["path"])
        except Exception:
            pass
        file_store.pop(token, None)
        used_tokens.add(token)
        qr_path = os.path.join(UPLOAD_DIR, f"{token}_qr.png")
        if os.path.exists(qr_path):
            os.remove(qr_path)
    background_tasks.add_task(cleanup)
    return response

@app.get("/file/qr/{token}")
async def get_qr(token: str):
    qr_path = os.path.join(UPLOAD_DIR, f"{token}_qr.png")
    if not os.path.exists(qr_path):
        raise HTTPException(status_code=404, detail="QR code not found.")
    return FileResponse(qr_path, media_type="image/png")

@app.get("/api/status")
async def status():
    # For debugging: show number of files in store
    return {"active_files": len(file_store)}
