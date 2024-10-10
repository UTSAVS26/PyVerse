Here’s a concise `README.md` file for the database setup in your Flask project:

---

# Database Setup for Medicine Prediction System

This project uses **SQLite** as the database for managing user information such as registration and login. The database schema includes one table for users.

## Database Configuration

The Flask app is configured to use SQLite as follows:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
```

- The database file `database.db` is automatically created when the app runs for the first time.

### Tables

1. **User Table (`User`)**: This table stores information about registered users.

   - **Columns**:
     - `id`: Primary key, integer.
     - `name`: User's name, string (max 100 chars).
     - `email`: User's email, string (max 100 chars, unique).
     - `password`: Hashed password using bcrypt, string (max 100 chars).

## Running Migrations

To create the database and tables, the following code is used:

```python
with app.app_context():
    db.create_all()
```

This ensures the table is created if it doesn't exist.

---

## Dataset Files

In addition to user data, the app uses CSV datasets for symptom-based predictions and health recommendations.

### Datasets

1. **symptoms_df.csv**: Contains symptoms for different diseases.
2. **precautions_df.csv**: Lists precautions for managing diseases.
3. **description.csv**: Provides a brief description of diseases.
4. **medications.csv**: Lists medications associated with each disease.
5. **diets.csv**: Recommended diets for diseases.

These datasets are loaded using `pandas`:

```python
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")
```

---

## User Registration and Login

- **Registration**: Allows users to sign up using a name, email, and password.
- **Login**: Allows registered users to log in using their email and password.
- **Password Security**: Passwords are hashed using `bcrypt`.

---

## Running the App

To run the Flask app:

```bash
python main.py
```

This will create the `database.db` file and ensure all necessary tables and datasets are loaded.

