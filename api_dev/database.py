""" imports """
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

""" create_engine is used to establish connection to database 
    session_maker is like a fcatory that produce new Session object when called
                  Session are used to interact with database
    declarative_base is used to define base class from which our all model classes will inhert
"""

""" here sqlite is used as database for simplicity but one can use mysql ,postgress also 
    for using mysql following steps need to be done
    1.install pymysql (pip install pymysql)
    2.create a new schema using mysql workbench 
    3.mysql+pymysql://root:<your-password>@localhost:3306/<database-name> paste this instead of sqlite:///./test.db
"""
URL_DATABASE = "sqlite:///./test.db"


engine = create_engine(URL_DATABASE)

SessionLocal = sessionmaker(autocommit=False ,autoflush=False ,bind=engine)

Base = declarative_base()
