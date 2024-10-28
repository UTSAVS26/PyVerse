**This Project will be a guide to create a minimalistic API for book store management
using FastAPI.**

To get started with this Project, Read the following:

1. set up virtual environment using python -m venv <name-of-virtual-environment>
2. when environment is created , activate it using <name-of-virtual-environment>/Scripts/activate
3. install dependencies using command pip install -r requirements.txt
4. start coding ,since we have three files here

   1. database.py
   2. models.py
   3. main.py

   start coding in this order

5. after finishing code you can run your program with this command
   uvicorn main:app --reload
6. http://127.0.0.1:8000 you will get this URL,now you can check your api
7. You can Postman/curl for testing.
8. How to Check your api ? Follow this steps mentioned below
   http://127.0.0.1:8000/books/ GET request for retrieving all books
   http://127.0.0.1:8000/books/ POST request for adding a new book
   http://127.0.0.1:8000/books/1 GET request for retireving book with bookid = 1
   http://127.0.0.1:8000/books/1 DELETE request for deleting book with bookid = 1
