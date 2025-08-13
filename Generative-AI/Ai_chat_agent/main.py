from fastapi import FastAPI, Request
from pydantic import BaseModel
from agents.intent_classifier import IntentClassifierAgent
from agents.router import RoutingAgent
from agents.notifier import NotifyAgent
from db.database import engine
from db.models import Base, MessageLog
from db.database import SessionLocal
from utils.logger import logger

import asyncio

app = FastAPI(title="AI Multi-Agent Chat Support System")

class ChatRequest(BaseModel):
    user: str
    message: str

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.post("/chat/")
async def chat(req: ChatRequest):
    intent = IntentClassifierAgent.classify(req.message)
    logger.info(f"Intent classified as: {intent}")

    response = await RoutingAgent.route(intent, req.user, req.message)

    async with SessionLocal() as session:
        msg_log = MessageLog(
            user=req.user,
            intent=intent,
            message=req.message,
            response=response
        )
        session.add(msg_log)
        await session.commit()

    asyncio.create_task(NotifyAgent.send(req.user, response))
    return {"intent": intent, "response": response}
