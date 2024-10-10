from pydantic import BaseModel, EmailStr

#register
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str

#update userinfo
class UserUpdate(BaseModel):
    name: str
    password: str

#delete
class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str
    role:str

class TokenData(BaseModel):
    email: str | None = None
