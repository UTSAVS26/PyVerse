""" imports """
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

""" create_engine is used to establish connection to database 
    sessionmaker is like a factory that produces new Session objects when called
    Session is used to interact with the database
    declarative_base is used to define a base class from which our model classes will inherit
"""

# Use SQLite for simplicity, but MySQL/PostgreSQL can also be used
URL_DATABASE = "sqlite:///./test.db"

engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Bug Fix: Ensure database tables are created when the engine is initialized
Base.metadata.create_all(bind=engine)
