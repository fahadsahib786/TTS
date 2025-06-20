"""
API routes for VoiceAI TTS Server
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
import json

from database import User, UsageLog, UserSession, get_db
from auth import auth_handler, create_user_session, refresh_access_token, logout_user

router = APIRouter()
security = HTTPBearer()

# --- Pydantic Models ---
class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    monthly_char_limit: int = 10000
    is_admin: bool = False
    expires_at: Optional[datetime] = None

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None
    full_name: Optional[str] = None
    monthly_char_limit: Optional[int] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    expires_at: Optional[datetime] = None

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    monthly_char_limit: int
    chars_used_current_month: int
    expires_at: Optional[datetime]
    created_at: datetime

class UsageLogResponse(BaseModel):
    id: int
    characters_used: int
    text_content: Optional[str]
    voice_mode: Optional[str]
    voice_file: Optional[str]
    generation_time: Optional[float]
    created_at: datetime

# --- Authentication Routes ---
@router.post("/auth/login")
async def login(
    user_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """Login user and create session"""
    user = db.query(User).filter(User.email == user_data.email).first()
    
    if not user or not user.check_password(user_data.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    
    if user.is_expired():
        raise HTTPException(status_code=403, detail="Account has expired")
    
    return await create_user_session(user, db, request)

@router.post("/auth/refresh")
async def refresh_token(
    request: Request,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token"""
    refresh_token = request.headers.get("X-Refresh-Token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="Refresh token is required")
    
    return await refresh_access_token(refresh_token, db, request)

@router.post("/auth/logout")
async def logout(
    all_sessions: bool = False,
    current_user: User = Depends(auth_handler.get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Logout user and invalidate session(s)"""
    current_token = credentials.credentials if credentials else None
    await logout_user(current_user, db, current_token, all_sessions)
    return {"message": "Successfully logged out"}

# --- User Management Routes (Admin Only) ---
@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(auth_handler.get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Create new user (admin only)"""
    # Check if email or username already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create new user
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=User.hash_password(user_data.password),
        full_name=user_data.full_name,
        is_admin=user_data.is_admin,
        monthly_char_limit=user_data.monthly_char_limit,
        expires_at=user_data.expires_at
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(auth_handler.get_current_admin_user),
    db: Session = Depends(get_db)
):
    """List all users (admin only)"""
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(auth_handler.get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get user details (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(auth_handler.get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Update user details (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update user fields
    if user_data.email is not None:
        existing = db.query(User).filter(User.email == user_data.email, User.id != user_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        user.email = user_data.email
    
    if user_data.username is not None:
        existing = db.query(User).filter(User.username == user_data.username, User.id != user_id).first()
        if existing:
            raise HTTPException(status_code=400, detail="Username already taken")
        user.username = user_data.username
    
    if user_data.password is not None:
        user.hashed_password = User.hash_password(user_data.password)
    
    if user_data.full_name is not None:
        user.full_name = user_data.full_name
    
    if user_data.monthly_char_limit is not None:
        user.monthly_char_limit = user_data.monthly_char_limit
    
    if user_data.is_active is not None:
        user.is_active = user_data.is_active
    
    if user_data.is_admin is not None:
        user.is_admin = user_data.is_admin
    
    if user_data.expires_at is not None:
        user.expires_at = user_data.expires_at
    
    db.commit()
    db.refresh(user)
    return user

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(auth_handler.get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    db.delete(user)
    db.commit()
    
    return {"message": "User deleted successfully"}

# --- Admin Statistics Routes ---
@router.get("/stats")
async def get_system_stats(
    current_user: User = Depends(auth_handler.get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get system-wide statistics (admin only)"""
    # Get total characters generated
    total_characters = db.query(func.sum(UsageLog.characters_used)).scalar() or 0
    
    # Get active users today
    today = datetime.utcnow().date()
    active_users_today = db.query(func.count(func.distinct(UsageLog.user_id))).filter(
        func.date(UsageLog.created_at) == today
    ).scalar() or 0
    
    # Get average generation time
    avg_gen_time = db.query(func.avg(UsageLog.generation_time)).filter(
        UsageLog.generation_time.isnot(None)
    ).scalar() or 0
    
    return {
        "total_characters": total_characters,
        "active_users_today": active_users_today,
        "average_generation_time": round(float(avg_gen_time), 2)
    }

# --- Usage Statistics Routes ---
@router.get("/users/{user_id}/usage", response_model=List[UsageLogResponse])
async def get_user_usage(
    user_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    """Get user usage statistics"""
    # Only admin can view other users' usage
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this user's usage")
    
    query = db.query(UsageLog).filter(UsageLog.user_id == user_id)
    
    if start_date:
        query = query.filter(UsageLog.created_at >= start_date)
    if end_date:
        query = query.filter(UsageLog.created_at <= end_date)
    
    usage_logs = query.order_by(UsageLog.created_at.desc()).all()
    return usage_logs

@router.get("/users/{user_id}/usage/summary")
async def get_user_usage_summary(
    user_id: int,
    current_user: User = Depends(auth_handler.get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get user usage summary statistics (admin only)"""
    # Get total characters used
    total_chars = db.query(func.sum(UsageLog.characters_used)).filter(UsageLog.user_id == user_id).scalar() or 0
    
    # Get current month usage
    start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    current_month_chars = db.query(func.sum(UsageLog.characters_used)).filter(
        UsageLog.user_id == user_id,
        UsageLog.created_at >= start_of_month
    ).scalar() or 0
    
    # Get average generation time
    avg_gen_time = db.query(func.avg(UsageLog.generation_time)).filter(
        UsageLog.user_id == user_id,
        UsageLog.generation_time.isnot(None)
    ).scalar() or 0
    
    return {
        "total_characters_used": total_chars,
        "current_month_characters": current_month_chars,
        "average_generation_time": round(avg_gen_time, 2)
    }

# --- Current User Routes ---
@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(auth_handler.get_current_user)
):
    """Get current user information"""
    return current_user

@router.get("/users/me/usage", response_model=List[UsageLogResponse])
async def get_current_user_usage(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's usage statistics"""
    return await get_user_usage(current_user.id, start_date, end_date, current_user, db)

@router.get("/users/me/usage/summary")
async def get_current_user_usage_summary(
    current_user: User = Depends(auth_handler.get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's usage summary"""
    return await get_user_usage_summary(current_user.id, current_user, db)

# --- Configuration Management Routes ---
@router.get("/config")
async def get_config(
    current_user: User = Depends(auth_handler.get_current_admin_user)
):
    """Get system configuration (admin only)"""
    # Return basic configuration settings that can be modified
    return {
        "max_text_length": 5000,
        "max_concurrent_requests": 5,
        "default_character_limit": 10000,
        "max_file_size_mb": 50,
        "session_timeout_hours": 24,
        "enable_user_registration": False
    }

@router.post("/config")
async def update_config(
    config_data: dict,
    current_user: User = Depends(auth_handler.get_current_admin_user)
):
    """Update system configuration (admin only)"""
    # In a real implementation, you would save these to a config file or database
    # For now, just return success
    return {"message": "Configuration updated successfully"}
