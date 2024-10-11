""" imports """
from sqlalchemy import Boolean ,Column ,Integer ,String ,DateTime
from database import Base

""" Column: as the name suggest it is used to create new column in the database 
    Boolean ,Integer ,String ,Datetime :- these are the datatype which will be stored in the database
"""

"""class Book will be table inside the database and it is inheriting from class Base 
   __tablename__ : name of the table
   id ,title ,author .. these are the columns  inside table
"""
class Book(Base):
    __tablename__ = 'books'
    
    id           = Column(Integer ,primary_key=True ,index=True)
    title        = Column(String(50) ,unique=True)
    author       = Column(String(50) ,unique=True)
    published_at = Column(DateTime)
    publication  = Column(String(100))
    pages        = Column(Integer)

