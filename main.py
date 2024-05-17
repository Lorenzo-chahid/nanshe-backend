import os
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    ForeignKey,
    DateTime,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import openai
from datetime import datetime
import io
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs


# Utiliser la variable d'environnement DATABASE_URL si elle est définie, sinon utiliser SQLite
ELEVENLABS_API_KEY = "xxxxxxxxxxxx"
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

templates = Jinja2Templates(directory="templates")
client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI()

# Configurer la clé API OpenAI à partir des variables d'environnement
openai.api_key = os.getenv("OPENAI_API_KEY")

# CORS configuration
origins = [
    "*",
    "http://localhost:3000",
    "http://localhost:3000/*",
    "https://nanshe-frontend.onrender.com",
    "https://nanshe-frontend.onrender.com/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Ajout des nouvelles colonnes dans les modèles SQLAlchemy
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    profile_image = Column(String, nullable=True)
    premium = Column(Boolean, default=False)
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
    profile_image = Column(String, nullable=True)
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
    profile_image: str = None
    premium: bool = False


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
    profile_image: str = None
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


def text_to_speech_stream(text: str) -> io.BytesIO:
    # Perform the text-to-speech conversion
    print("HEEEEREEE ::: ", text)
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",  # Example voice ID
        text=text,
        output_format="mp3_22050_32",
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Create a BytesIO object to hold the audio data in memory
    audio_stream = io.BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)  # Rewind the stream to the beginning
    return audio_stream


class SpeechRequest(BaseModel):
    text: str


@app.post("/text-to-speech/")
async def text_to_speech(request: SpeechRequest):
    text = request.text
    audio_stream = text_to_speech_stream(text)
    print("HEERE MANEKE ::: ", text)
    return {"message": audio_stream}


@app.post("/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(
        username=user.username,
        email=user.email,
        password=user.password,
        profile_image=user.profile_image,
        premium=user.premium,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/avatars/", response_model=AvatarCreate)
def create_avatar(avatar: AvatarCreate, db: Session = Depends(get_db)):
    db_avatar = Avatar(
        first_name=avatar.first_name,
        last_name=avatar.last_name,
        age=avatar.age,
        personality=avatar.personality,
        traits=avatar.traits,
        writing=avatar.writing,
        profile_image=avatar.profile_image,
        user_id=avatar.user_id,
    )
    db.add(db_avatar)
    db.commit()
    db.refresh(db_avatar)
    return db_avatar


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
    return {"message": "Login successful", "user_id": db_user.id}


@app.get("/avatars/{user_id}")
def get_avatars(user_id: int, db: Session = Depends(get_db)):
    avatars = db.query(Avatar).filter(Avatar.user_id == user_id).all()
    return avatars


@app.get("/conversations/{avatar_id}")
def get_conversations(avatar_id: int, db: Session = Depends(get_db)):
    conversations = (
        db.query(Conversation, Avatar.profile_image, User.profile_image)
        .join(Avatar, Avatar.id == Conversation.avatar_id)
        .join(User, User.id == Avatar.user_id)
        .filter(Conversation.avatar_id == avatar_id)
        .order_by(Conversation.created_at.asc())
        .all()
    )
    result = []
    for conv, avatar_img, user_img in conversations:
        result.append(
            {
                "id": conv.id,
                "avatar_id": conv.avatar_id,
                "user_message": conv.user_message,
                "avatar_response": conv.avatar_response,
                "created_at": conv.created_at,
                "avatar_image": avatar_img,
                "user_image": user_img,
            }
        )
    return result


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


##### ADMIN ############


@app.get("/admin/users/", response_class=HTMLResponse)
async def admin_list_users(request: Request, db: Session = Depends(get_db)):
    users = db.query(User).all()
    return templates.TemplateResponse(
        "admin_users.html", {"request": request, "users": users}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
