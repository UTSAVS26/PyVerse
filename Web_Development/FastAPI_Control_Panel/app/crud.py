from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


#Read User
def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

#Create User
def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = pwd_context.hash(user.password)
    db_user = models.User(name=user.name, email=user.email, hashed_password=hashed_password, role=user.role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

#Update User
def update_user(db: Session, user: models.User, user_update: schemas.UserUpdate):
    user.name = user_update.name
    user.hashed_password = pwd_context.hash(user_update.password)
    db.commit()
    db.refresh(user)
    return user

#Read All Users
def get_users(db: Session):
    return db.query(models.User).all()

#Read User by ID
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

#Delete User by ID
def delete_user(db: Session, user_id: int):
    user = get_user(db, user_id)
    if user:
        db.delete(user)
        db.commit()
        return True
    return False
