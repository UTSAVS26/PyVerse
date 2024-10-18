""" imports """
from sqlalchemy import Boolean, Column, Integer, String, DateTime
from database import Base

""" 
Book model for the 'books' table in the database
"""
class Book(Base):
    __tablename__ = 'books'
    
    id           = Column(Integer, primary_key=True, index=True)
    title        = Column(String(100), unique=True)  # Fixed: Increased string length for title
    author       = Column(String(100))               # Fixed: Removed unique constraint on author
    published_at = Column(DateTime)
    publication  = Column(String(100))
    pages        = Column(Integer)
