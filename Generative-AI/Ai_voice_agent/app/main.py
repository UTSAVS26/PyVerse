from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.call_logic import handle_inbound_call, handle_outbound_call
from app.utils import setup_logging
import os

app = FastAPI(title="AI Voice Agent PoC")

setup_logging()

@app.get("/")
async def root():
    # Serve your frontend index.html here from static folder explicitly
    index_path = os.path.join("static", "index.html")
    return HTMLResponse(content=open(index_path).read(), status_code=200)

@app.post("/inbound-call")
async def inbound_call(data: dict):
    try:
        result = await handle_inbound_call(data)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/outbound-call")
async def outbound_call(data: dict):
    try:
        result = await handle_outbound_call(data)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio")
async def get_audio():
    audio_path = "output/output.mp3"
    return FileResponse(audio_path, media_type="audio/mpeg")

# Mount static files under /static path
app.mount("/static", StaticFiles(directory="static"), name="static")
