""" imports """
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Annotated, List
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime

""" Instantiating an object of FastAPI class """
app = FastAPI()
models.Base.metadata.create_all(bind=engine)

""" Pydantic model for data validation """
class BookBase(BaseModel):
    title: str
    author: str
    published_at: datetime
    publication: str
    pages: int

    class Config:
        orm_mode = True

""" Database session lifecycle management """
def get_database():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()

database_dependency = Annotated[Session, Depends(get_database)]

""" API Endpoints """

# Create a new book
@app.post('/books/', status_code=status.HTTP_201_CREATED, response_model=BookBase)
async def create_book(book: BookBase, db: database_dependency):
    """ 
    Create a new book entry, handle duplicate entries
    """
    try:
        new_book = models.Book(**book.dict())
        db.add(new_book)
        db.commit()
        db.refresh(new_book)
        return new_book
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Book with this title or author already exists")

# Get all books
@app.get('/books/', response_model=List[BookBase])
async def get_all_books(db: database_dependency):
    """ Retrieve all books from the database """
    books = db.query(models.Book).all()
    return books

# Get a specific book by ID
@app.get('/books/{book_id}', response_model=BookBase)
async def get_book(book_id: int, db: database_dependency):
    """ Retrieve a book with the given book ID """
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if book:
        return book
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")

# Update a book by ID
@app.patch('/books/{book_id}', response_model=BookBase)
async def update_book(book_id: int, book_data: BookBase, db: database_dependency):
    """ Update book details by ID """
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")

    for key, value in book_data.dict(exclude_unset=True).items():
        setattr(book, key, value)

    db.commit()
    db.refresh(book)
    return book

# Delete a book by ID
@app.delete('/books/{book_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(book_id: int, db: database_dependency):
    """ Delete a book with the given ID """
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")

    db.delete(book)
    db.commit()
    return None  # Fixed: Return None to respect 204 No Content status code
