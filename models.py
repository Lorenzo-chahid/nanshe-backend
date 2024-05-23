# backend/models.py
from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

user_friends = Table(
    "user_friends",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("friend_id", Integer, ForeignKey("users.id"), primary_key=True),
)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    profile_image = Column(String, nullable=True)
    premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    avatars = relationship("Avatar", back_populates="owner")
    friends = relationship(
        "User",
        secondary=user_friends,
        primaryjoin=id == user_friends.c.user_id,
        secondaryjoin=id == user_friends.c.friend_id,
        backref="user_friends",
    )
    liked_messages = relationship("LikedMessage", back_populates="user")


class Avatar(Base):
    __tablename__ = "avatars"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True, nullable=True)
    age = Column(Integer)
    gender = Column(String)
    personality = Column(String, nullable=True)
    traits = Column(String, nullable=True)
    writing = Column(String)
    profile_image = Column(String, nullable=True)
    eye_color = Column(String, nullable=True)
    hair_color = Column(String, nullable=True)
    weight = Column(Integer, nullable=True)
    bust_size = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="avatars")
    conversations = relationship("Conversation", back_populates="avatar")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    avatar_id = Column(Integer, ForeignKey("avatars.id"))
    user_message = Column(String)
    avatar_response = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    avatar = relationship("Avatar", back_populates="conversations")


class LikedMessage(Base):
    __tablename__ = "liked_messages"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message_id = Column(Integer, ForeignKey("conversations.id"))
    user = relationship("User", back_populates="liked_messages")
