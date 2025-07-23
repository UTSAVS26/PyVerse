from .models import CallLog
from .db import SessionLocal
from .stt import transcribe
from .intent import extract_intent
import asyncio

def log_call(caller, direction, transcript, intent):
    db = SessionLocal()
    call = CallLog(
        caller=caller,
        direction=direction,
        transcript=transcript,
        intent=intent
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    db.close()
    return call

async def handle_call(caller: str, audio_url: str, direction: str):
    transcript = await transcribe(audio_url)
    intent = extract_intent(transcript)
    return log_call(caller, direction, transcript, intent)

async def handle_inbound_call(caller: str, audio_url: str):
    return await handle_call(caller, audio_url, direction="inbound")

async def handle_outbound_call(caller: str, audio_url: str):
    return await handle_call(caller, audio_url, direction="outbound")
