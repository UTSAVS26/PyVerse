from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import subprocess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Database connection setup
def get_db_connection():
    conn = sqlite3.connect('playground.db')
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS snippets (id INTEGER PRIMARY KEY, code TEXT)''')
        conn.commit()
    finally:
        conn.close()

# Ensure CORS is set up to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins if needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

@app.on_event("startup")
async def startup_event():
    initialize_database()

@app.post("/execute")
async def execute_code(code_request: CodeRequest):
    try:
        # Write the code to a temporary Python file
        with open("temp_code.py", "w") as code_file:
            code_file.write(code_request.code)

        # Execute the Python file and capture the output
        result = subprocess.run(
            ["python", "temp_code.py"],
            capture_output=True,
            text=True,
            timeout=5  # Set a timeout to prevent long-running code
        )

        return {"output": result.stdout, "errors": result.stderr}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Code execution timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_snippet")
async def save_snippet(code_request: CodeRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO snippets (code) VALUES (?)", (code_request.code,))
        conn.commit()
        return {"message": "Snippet saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/snippets")
async def get_snippets():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM snippets")
        snippets = cursor.fetchall()
        return {"snippets": [dict(snippet) for snippet in snippets]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# Static file serving
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
