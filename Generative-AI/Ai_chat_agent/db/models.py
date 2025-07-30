from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    user = Column(String, index=True)
    message = Column(Text)
    status = Column(String, default="open")
    created_at = Column(DateTime, server_default=func.now())

class MessageLog(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    user = Column(String, index=True)
    intent = Column(String)
    message = Column(Text)
    response = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
