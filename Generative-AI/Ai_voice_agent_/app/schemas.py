from pydantic import BaseModel
from datetime import datetime

class CallLogSchema(BaseModel):
    caller: str
    direction: str
    transcript: str
    intent: str
    timestamp: datetime

    class Config:
        orm_mode = True
