from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from ..security import create_token, current_user, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)


class LoginIn(BaseModel):
    username: str
    password: str


def _user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "displayName": u.display_name, "createdAt": u.created_at.isoformat()}


@router.post("/register", status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    username = body.username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,50}", username):
        raise HTTPException(422, "Username may contain letters, digits, '.', '_' and '-' only.")
    if db.scalar(select(User).where(User.username == username)):
        raise HTTPException(409, "That username is taken.")
    user = User(username=username, display_name=(body.display_name or username).strip(),
                password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return {"token": create_token(user), "user": _user_out(user)}


@router.post("/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.username == body.username.strip().lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect username or password.")
    return {"token": create_token(user), "user": _user_out(user)}


@router.get("/me")
def me(user: User = Depends(current_user)):
    return _user_out(user)
