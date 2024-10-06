""" imports """
from fastapi import FastAPI ,HTTPException ,Depends ,status
from pydantic import BaseModel
from typing import Annotated, List
import models
from database import engine ,SessionLocal
from sqlalchemy.orm import Session
from datetime import datetime

""" instantiating an object of FastAPI class"""
app = FastAPI()
models.Base.metadata.create_all(bind=engine)

""" Defining class BookBase which is inheriting from pydantic BaseModel 
    pydantic model are used to define the structure and data validation rules for 
    incoming and outgoing and data
"""
class BookBase(BaseModel):
    title:str
    author:str
    published_at:datetime
    publication:str
    pages:int
    
""" this method monitors the life cycle of database session."""
def get_database():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()

database_dependency = Annotated[Session ,Depends(get_database)]

""" API endpoints """
@app.post('/books/',status_code=status.HTTP_201_CREATED ,response_model=BookBase)
async def create_book(book:BookBase ,db:database_dependency):
    """
    Create a new book entry in the database.

    Parameters:
    - book (BookBase): An object that contains the data required to create a book. This is validated by the BookBase Pydantic model.
    - db (database_dependency): A database session object used to interact with the database.

    Returns:
    - BookBase: The newly created book object, returned as a response in the format defined by the BookBase model.

    Process:
    1. Receives book data from the request.
    2. Converts the Pydantic `BookBase` object into a dictionary.
    3. Creates a new `Book` model instance with the data and adds it to the database session.
    4. Commits the changes to persist the new book in the database.
    5. Refreshes the instance to update it with any changes made by the database (e.g., the auto-generated `id`).
    6. Returns the newly created book to the client.
    """
    new_book = models.Book(**book.dict())
    db.add(new_book)
    db.commit()
    db.refresh(new_book)
    return new_book

@app.get('/books/',response_model=List[BookBase])
async def get_all_books(db:database_dependency):
    """ 
    Retrieve all book from database
    
    parameters: 
    - db (database_dependency): A database session object used to interact with the database.
    
    Returns:
    - List[BookBase]: A list of all book objects in the database, returned in the format defined by the BookBase model.

    Process:
    1. Queries the database to retrieve all book entries using SQLAlchemy.
    2. Returns the list of book objects, which FastAPI will serialize into JSON format.
    """
    books = db.query(models.Book).all()
    return books

@app.get('/books/{book_id}')
async def get_book(book_id:int ,db:database_dependency):
    """ 
    Retrieve Book with given book id
    
    parameters:
    - book_id : id of book
    - db (database_dependency): A database session object used to interact with the database
    
    Returns:
    - book object if it exist
    
    Process:
    1. Queries the database to retrive book with given book_id using SQLAlchemy
    2. Checks if book exist then return it and if not then raise HTTPException
    """
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if book:
        return book
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    
@app.patch('/books/{book_id}', response_model=BookBase)
async def update_book(book_id: int, book_data: BookBase, db: database_dependency):
    """
    Update the details of a specific book identified by its ID.

    Parameters:
    - book_id (int): The ID of the book to be updated.
    - book_data (BookBase): An object containing the updated book data.
    - db (database_dependency): A database session object used to interact with the database.

    Returns:
    - BookBase: The updated book object.

    Process:
    1. Queries the database to find the book with the given book_id.
    2. Checks if the book exists; if not, raises an HTTPException with a 404 status code.
    3. Updates the book's attributes with the provided data, excluding any unset fields.
    4. Commits the changes to the database and refreshes the book object to reflect the latest data.
    5. Returns the updated book object.
    """
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")

    
    for key, value in book_data.dict(exclude_unset=True).items():
        setattr(book, key, value)

    db.commit()
    db.refresh(book)
    return book


@app.delete('/books/{book_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_book(book_id: int, db: database_dependency):
    """ 
    Delete the book with the given ID from the database.

    Parameters:
    - book_id (int): The ID of the book to be deleted.
    - db (database_dependency): A database session object used to interact with the database.

    Returns:
    - None: Returns a 204 No Content response upon successful deletion.

    Process:
    1. Queries the database to find the book with the given book_id.
    2. Checks if the book exists; if not, raises an HTTPException with a 404 status code.
    3. Deletes the book from the database and commits the changes.
    4. Returns a success message
    """
    book = db.query(models.Book).filter(models.Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    
    db.delete(book)
    db.commit()
    return {"message": "Book deleted successfully"}