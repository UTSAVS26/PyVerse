from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from .db import Base

class CallLog(Base):
    __tablename__ = "call_logs"
    id = Column(Integer, primary_key=True, index=True)
    caller = Column(String)
    direction = Column(String)  # inbound or outbound
    transcript = Column(Text)
    intent = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
