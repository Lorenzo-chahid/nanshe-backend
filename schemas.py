# backend/schemas.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    profile_image: Optional[str] = None
    premium: bool = False


class UserLogin(BaseModel):
    login: str
    password: str


class AvatarCreate(BaseModel):
    first_name: str
    last_name: Optional[str] = None
    age: int
    gender: Optional[str] = None
    personality: Optional[str] = None
    traits: Optional[str] = None
    writing: str
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None
    weight: Optional[int] = None
    bust_size: Optional[str] = None
    profile_image: Optional[str] = None
    user_id: int

    class Config:
        from_attribute = True


class Message(BaseModel):
    avatar_id: int
    user_message: str


class SpeechRequest(BaseModel):
    text: str
