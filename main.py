import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import openai
from datetime import datetime

# Utiliser la variable d'environnement DATABASE_URL si elle est définie, sinon utiliser SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

# Configurer la clé API OpenAI à partir des variables d'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")

# CORS configuration
origins = [
    "http://localhost:3000",
    "https://nanshe-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    avatars = relationship("Avatar", back_populates="owner")


class Avatar(Base):
    __tablename__ = "avatars"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True, nullable=True)
    age = Column(Integer)
    personality = Column(String, nullable=True)
    traits = Column(String, nullable=True)
    writing = Column(String)
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


Base.metadata.create_all(bind=engine)


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    login: str
    password: str


class AvatarCreate(BaseModel):
    first_name: str
    last_name: str = None
    age: int
    personality: str = None
    traits: str = None
    writing: str
    user_id: int


class Message(BaseModel):
    avatar_id: int
    user_message: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(username=user.username, email=user.email, password=user.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/login/")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = (
        db.query(User)
        .filter(
            (User.email == user.login) | (User.username == user.login),
            User.password == user.password,
        )
        .first()
    )
    if not db_user:
        raise HTTPException(status_code=400, detail="Invalid login or password")
    return {"message": "Login successful"}


@app.post("/avatars/", response_model=AvatarCreate)
def create_avatar(avatar: AvatarCreate, db: Session = Depends(get_db)):
    db_avatar = Avatar(**avatar.dict())
    db.add(db_avatar)
    db.commit()
    db.refresh(db_avatar)
    return db_avatar


@app.get("/avatars/{user_id}")
def get_avatars(user_id: int, db: Session = Depends(get_db)):
    avatars = db.query(Avatar).filter(Avatar.user_id == user_id).all()
    return avatars


@app.post("/chat/")
async def chat(message: Message, db: Session = Depends(get_db)):
    try:
        # Fetch avatar details
        avatar = db.query(Avatar).filter(Avatar.id == message.avatar_id).first()
        if not avatar:
            raise HTTPException(status_code=404, detail="Avatar not found")

        # Fetch conversation history
        conversation_history = (
            db.query(Conversation)
            .filter(Conversation.avatar_id == message.avatar_id)
            .all()
        )

        # Construct the prompt for OpenAI
        messages = [
            {
                "role": "system",
                "content": f"You are an AI playing the role of {avatar.first_name} {avatar.last_name}, a {avatar.age} years old avatar with a {avatar.personality} personality and {avatar.traits} traits. The writing style is {avatar.writing}.",
            }
        ]

        for c in conversation_history:
            messages.append({"role": "user", "content": c.user_message})
            messages.append({"role": "assistant", "content": c.avatar_response})

        messages.append({"role": "user", "content": message.user_message})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=150
        )

        ai_response = response.choices[0].message["content"].strip()

        # Save the current conversation
        new_conversation = Conversation(
            avatar_id=message.avatar_id,
            user_message=message.user_message,
            avatar_response=ai_response,
        )
        db.add(new_conversation)
        db.commit()

        return {"avatar_response": ai_response}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
