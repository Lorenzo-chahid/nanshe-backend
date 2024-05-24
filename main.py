# backend/main.py
import os
import io
import requests
from sqlalchemy import or_
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import openai

from models import (
    AvatarSkill,
    Base,
    User,
    Avatar,
    Conversation,
    LikedMessage,
    user_friends,
)
from schemas import UserCreate, UserLogin, AvatarCreate, Message, SpeechRequest
from database import engine, get_db
from ia_analyzer import ImageAnalyzer

openai.api_key = os.getenv("OPENAI_API_KEY")

# Création des tables dans la base de données
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Configuration des templates Jinja2
templates = Jinja2Templates(directory="templates")

# Configuration CORS
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

image_analyzer = ImageAnalyzer()


def calculate_experience_for_next_level(level):
    return 50 * (2 ** (level // 5))


def update_experience(db_avatar: Avatar, xp: float, db: Session):
    db_avatar.experience += xp
    level_up = False

    while db_avatar.experience >= calculate_experience_for_next_level(db_avatar.level):
        db_avatar.experience -= calculate_experience_for_next_level(db_avatar.level)
        db_avatar.level += 1
        level_up = True

        if db_avatar.level >= 5 and db_avatar.level < 15:
            db_avatar.relationship_status = "connaissance"
        elif db_avatar.level >= 15 and db_avatar.level < 25:
            db_avatar.relationship_status = "ami"
        elif db_avatar.level >= 25:
            db_avatar.relationship_status = "sentimental"

    db.commit()
    db.refresh(db_avatar)

    return level_up


def upgrade_skill(db: Session, avatar_id: int, skill_name: str):
    skill = (
        db.query(AvatarSkill)
        .filter(
            AvatarSkill.avatar_id == avatar_id, AvatarSkill.skill_name == skill_name
        )
        .first()
    )
    if not skill:
        skill = AvatarSkill(avatar_id=avatar_id, skill_name=skill_name, level=1)
        db.add(skill)
    else:
        skill.level += 1

    db.commit()
    db.refresh(skill)
    return skill


def print_users_table(db: Session):
    users = db.query(User).all()
    print("ID | Username | Email | Password | Profile Image | Premium | Created At")
    print("---------------------------------------------------------------")
    for user in users:
        print(
            f"{user.id} | {user.username} | {user.email} | {user.password} | {user.profile_image} | {user.premium} | {user.created_at}"
        )


def print_avatar_table(db: Session):
    avatar = db.query(Avatar).all()
    print("ID | First Name | Skills | Experience | Profile Image")
    print("---------------------------------------------------------------")
    for av in avatar:
        print(
            f"{av.id} | {av.first_name} | {av.skills} | {av.experience} | {av.profile_image} | "
        )


@app.post("/users/", response_model=UserCreate)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = (
        db.query(User)
        .filter(or_(User.email == user.email, User.username == user.username))
        .first()
    )
    if db_user:
        raise HTTPException(
            status_code=400, detail="Email or username already registered"
        )
    new_user = User(
        username=user.username,
        email=user.email,
        password=user.password,
        profile_image=user.profile_image,
        premium=False,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Print the users table after adding the new user
    print_users_table(db)

    return new_user


@app.post("/avatars/", response_model=AvatarCreate)
def create_avatar(avatar: AvatarCreate, db: Session = Depends(get_db)):
    db_avatar = Avatar(
        first_name=avatar.first_name,
        last_name=avatar.last_name,
        age=avatar.age,
        gender=avatar.gender,
        personality=avatar.personality,
        traits=avatar.traits,
        writing=avatar.writing,
        profile_image=avatar.profile_image,
        eye_color=avatar.eye_color,
        hair_color=avatar.hair_color,
        weight=avatar.weight,
        bust_size=avatar.bust_size,
        user_id=avatar.user_id,
    )
    db.add(db_avatar)
    db.commit()
    db.refresh(db_avatar)
    return db_avatar


@app.put("/avatar/{avatar_id}", response_model=AvatarCreate)
def update_avatar(
    avatar_id: int,
    avatar: AvatarCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    print_avatar_table(db)
    db_avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
    if not db_avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    db_avatar.first_name = avatar.first_name
    db_avatar.last_name = avatar.last_name
    db_avatar.age = avatar.age
    db_avatar.gender = avatar.gender
    db_avatar.personality = avatar.personality
    db_avatar.traits = avatar.traits
    db_avatar.writing = avatar.writing
    db_avatar.profile_image = avatar.profile_image
    db_avatar.eye_color = avatar.eye_color
    db_avatar.hair_color = avatar.hair_color
    db_avatar.weight = avatar.weight
    db_avatar.bust_size = avatar.bust_size
    db_avatar.user_id = avatar.user_id

    db.commit()
    db.refresh(db_avatar)

    if db_avatar.profile_image:
        background_tasks.add_task(
            update_avatar_physical_description,
            db_avatar.profile_image,
            db_avatar.id,
            db,
        )

    return db_avatar


@app.post("/avatar/{avatar_id}/experience")
def add_experience(avatar_id: int, xp: int, db: Session = Depends(get_db)):
    db_avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
    if not db_avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    level_up = update_experience(db_avatar, xp, db)
    return {"message": "Experience added", "level_up": level_up}


@app.post("/avatar/{avatar_id}/skill")
def upgrade_avatar_skill(
    avatar_id: int, skill_name: str, db: Session = Depends(get_db)
):
    skill = upgrade_skill(db, avatar_id, skill_name)
    return {"message": "Skill upgraded", "skill": skill}


def update_avatar_physical_description(
    profile_image_url: str, avatar_id: int, db: Session
):
    try:
        description = image_analyzer.analyze_image(profile_image_url)
        print("HERE :: ", description)
        db_avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
        if db_avatar:
            db_avatar.physical_description = description
            db.commit()
            db.refresh(db_avatar)
    except Exception as e:
        print(f"Error updating avatar physical description: {e}")


@app.delete("/avatar/{avatar_id}", response_model=dict)
def delete_avatar(avatar_id: int, db: Session = Depends(get_db)):
    db_avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
    if not db_avatar:
        raise HTTPException(status_code=404, detail="Avatar not found")

    db.delete(db_avatar)
    db.commit()
    return {"message": "Avatar deleted successfully"}


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


@app.get("/avatar/{avatar_id}", response_model=AvatarCreate)
def get_avatar(avatar_id: int, db: Session = Depends(get_db)):
    try:
        avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
        if not avatar:
            print(f"Avatar with ID {avatar_id} not found.")
            raise HTTPException(status_code=404, detail="Avatar not found")
        print(f"Fetched avatar: {avatar}")
        return avatar
    except Exception as e:
        print(f"Error fetching avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{avatar_id}")
def get_conversations(avatar_id: int, db: Session = Depends(get_db)):
    avatar = db.query(Avatar).filter(Avatar.id == avatar_id).first()
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
                "content": (
                    f"You are an AI playing the role of {avatar.first_name} {avatar.last_name}, a {avatar.age} years old avatar "
                    f"with a {avatar.personality} personality and {avatar.traits} traits. The writing style is {avatar.writing}. "
                    "You must stay in character at all times and never reveal that you are an AI or a language model. "
                    "Respond as if you are the character in every situation."
                ),
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

        # Update experience for both receiving and sending messages
        update_experience(avatar, 0.5, db)  # Receiving message
        update_experience(avatar, 1, db)  # Sending message

        return {"avatar_response": ai_response}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-image/")
async def generate_image(request: Request):
    body = await request.json()
    physical = body.get("physical", {})
    gender = body.get("gender", "")

    if not physical or not gender:
        raise HTTPException(
            status_code=400, detail="Physical data and gender are required"
        )

    # Create a prompt based on the physical data and gender
    prompt = f"a nude woman"
    for k, v in physical.items():
        print("MANNEKE :: ", k, "------", v)

    url = "https://api.getimg.ai/v1/stable-diffusion-xl/text-to-image"
    payload = {
        "model": "stable-diffusion-xl-v1-0",
        "prompt": "une image photo réaliste d'une femme nue en train d'avoir un orgasme. sa poitrine fera du 90D",
        "output_format": "jpeg",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer key-1vj7bYgbfSeukCo7rqzlUsiIh4IPfGwBiq0Rplq6jNfWa4TkliByMg4rzLaMg65eW3LDu1UdT5yJhawCuSN7h6MpSSmEzCRA",
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to generate image"
        )

    try:
        response_json = response.json()
        image_data = response_json["image"]
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=500, detail="Invalid response from image generation API"
        )

    return {"image": image_data}


@app.get("/admin/users/", response_class=HTMLResponse)
async def admin_list_users(request: Request, db: Session = Depends(get_db)):
    users = db.query(User).all()
    return templates.TemplateResponse(
        "admin_users.html", {"request": request, "users": users}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
