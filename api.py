"""
API routes for VoiceAI TTS Server
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address
import json

from database import User, UsageLog, UserSession, LoginAttempt, get_db
from auth import auth_handler, create_user_session, refresh_access_token, logout_user

router = APIRouter()
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

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
    daily_char_limit: int = 1000
    per_request_char_limit: int = 500
    is_admin: bool = False
    expires_at: Optional[datetime] = None

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None
    full_name: Optional[str] = None
    monthly_char_limit: Optional[int] = None
    daily_char_limit: Optional[int] = None
    per_request_char_limit: Optional[int] = None
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
    daily_char_limit: int
    per_request_char_limit: int
    chars_used_current_month: int
    chars_used_today: int
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
@limiter.limit("5/minute")
async def login(
    user_data: UserLogin,
    request: Request = None,
    response: Response = None,
    db: Session = Depends(get_db)
):
    """Login user and create session with brute force protection"""
    import logging
    logger = logging.getLogger(__name__)
    
    ip_address = request.client.host
    user_agent = request.headers.get("user-agent")
    
    logger.info(f"[Login] Login attempt received:")
    logger.info(f"[Login] Email: {user_data.email}")
    logger.info(f"[Login] IP Address: {ip_address}")
    logger.info(f"[Login] User Agent: {user_agent}")
    logger.info(f"[Login] Password provided: {'Yes' if user_data.password else 'No'}")
    logger.info(f"[Login] Password length: {len(user_data.password) if user_data.password else 0}")

    # Check for IP-based blocking
    if LoginAttempt.is_ip_blocked(ip_address, db):
        logger.warning(f"[Login] IP {ip_address} is blocked due to too many failed attempts")
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts. Please try again later."
        )

    # Check for email-based blocking
    if LoginAttempt.is_email_blocked(user_data.email, db):
        logger.warning(f"[Login] Email {user_data.email} is blocked due to too many failed attempts")
        raise HTTPException(
            status_code=429,
            detail="Too many failed attempts for this email. Please try again later."
        )

    # Create login attempt record
    login_attempt = LoginAttempt(
        email=user_data.email,
        ip_address=ip_address,
        user_agent=user_agent,
        success=False
    )
    db.add(login_attempt)
    logger.info(f"[Login] Created login attempt record for {user_data.email}")
    
    # Look up user
    logger.info(f"[Login] Looking up user with email: {user_data.email}")
    user = db.query(User).filter(User.email == user_data.email).first()
    
    if not user:
        logger.warning(f"[Login] No user found with email: {user_data.email}")
        db.commit()  # Save the failed attempt
        raise HTTPException(status_code=401, detail="No account found with this email address")
    
    logger.info(f"[Login] User found: ID={user.id}, Username={user.username}, Active={user.is_active}")
    
    # Check password
    logger.info(f"[Login] Checking password for user {user.username}")
    password_valid = user.check_password(user_data.password)
    logger.info(f"[Login] Password validation result: {password_valid}")
    
    if not password_valid:
        logger.warning(f"[Login] Invalid password for user {user.username}")
        db.commit()  # Save the failed attempt
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    if not user.is_active:
        logger.warning(f"[Login] User {user.username} account is disabled")
        db.commit()  # Save the failed attempt
        raise HTTPException(status_code=403, detail="This account has been disabled. Please contact an administrator.")
    
    if user.is_expired():
        logger.warning(f"[Login] User {user.username} account is expired")
        db.commit()  # Save the failed attempt
        raise HTTPException(status_code=403, detail="This account has expired. Please contact an administrator to renew.")
    
    # Update login attempt as successful
    login_attempt.success = True
    db.commit()
    logger.info(f"[Login] Login successful for user {user.username}, creating session...")
    
    # Create user session
    try:
        session_result = await create_user_session(user, db, request, response)
        logger.info(f"[Login] Session created successfully for user {user.username}")
        logger.info(f"[Login] Access token created: {session_result['access_token'][:20] if session_result.get('access_token') else 'NONE'}...")
        logger.info(f"[Login] Refresh token created: {session_result['refresh_token'][:20] if session_result.get('refresh_token') else 'NONE'}...")
        logger.info(f"[Login] Token type: {session_result.get('token_type')}")
        logger.info(f"[Login] Expires in: {session_result.get('expires_in')} seconds")
        return session_result
    except Exception as e:
        logger.error(f"[Login] Error creating session for user {user.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create user session")

@router.post("/auth/refresh")
@limiter.limit("10/minute")
async def refresh_token(
    request: Request = None,
    response: Response = None,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token"""
    refresh_token = request.headers.get("X-Refresh-Token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="Refresh token is required")
    
    result = await refresh_access_token(refresh_token, db, request)
    
    # Set HTTP-only cookie for the new access token
    if response:
        is_secure = request.url.scheme == "https" if request else False
        access_cookie_max_age = 300 * 60  # ACCESS_TOKEN_EXPIRE_MINUTES * 60
        
        response.set_cookie(
            key="access_token",
            value=result["access_token"],
            max_age=access_cookie_max_age,
            httponly=True,
            secure=is_secure,
            samesite="lax",
            path="/",
            domain=None
        )
    
    return result

@router.post("/auth/logout")
async def logout(
    request: Request,
    response: Response,
    all_sessions: bool = False,
    db: Session = Depends(get_db)
):
    """Logout user and invalidate session(s)"""
    current_user = await auth_handler.get_current_user(request, db)
    
    # Get current token from request
    current_token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        current_token = auth_header[7:]
    elif request.cookies.get("access_token"):
        current_token = request.cookies.get("access_token")
    
    await logout_user(current_user, db, current_token, all_sessions)
    
    # Clear cookies with matching settings
    is_secure = request.url.scheme == "https"
    response.delete_cookie(
        key="access_token",
        httponly=True,
        secure=is_secure,
        samesite="lax",
        path="/",
        domain=None  # Let browser determine domain
    )
    response.delete_cookie(
        key="refresh_token",
        httponly=True,
        secure=is_secure,
        samesite="lax",
        path="/",
        domain=None  # Let browser determine domain
    )
    
    return {"message": "Successfully logged out"}

# --- User Management Routes (Admin Only) ---
@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Create new user (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
    # Validate password complexity
    is_valid, message = User.validate_password_complexity(user_data.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
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
        daily_char_limit=user_data.daily_char_limit,
        per_request_char_limit=user_data.per_request_char_limit,
        expires_at=user_data.expires_at
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    request: Request,
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """List all users (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Get user details (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    request: Request,
    db: Session = Depends(get_db)
):
    """Update user details (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
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
        # Validate password complexity
        is_valid, message = User.validate_password_complexity(user_data.password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        user.hashed_password = User.hash_password(user_data.password)
        
        # Invalidate all user sessions when password is changed
        db.query(UserSession).filter(UserSession.user_id == user.id).update({"is_active": False})
    
    if user_data.full_name is not None:
        user.full_name = user_data.full_name
    
    if user_data.monthly_char_limit is not None:
        user.monthly_char_limit = user_data.monthly_char_limit
    
    if user_data.daily_char_limit is not None:
        user.daily_char_limit = user_data.daily_char_limit
    
    if user_data.per_request_char_limit is not None:
        user.per_request_char_limit = user_data.per_request_char_limit
    
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
    request: Request,
    db: Session = Depends(get_db)
):
    """Delete user (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
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
    request: Request,
    db: Session = Depends(get_db)
):
    """Get system-wide statistics (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
    # Get total characters generated
    total_characters = db.query(func.sum(UsageLog.characters_used)).scalar() or 0
    
    # Get active users today
    today = datetime.now(timezone.utc).date()
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
    request: Request,
    db: Session = Depends(get_db),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get user usage statistics"""
    current_user = await auth_handler.get_current_user(request, db)
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
    request: Request,
    db: Session = Depends(get_db)
):
    """Get user usage summary statistics (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
    # Get total characters used
    total_chars = db.query(func.sum(UsageLog.characters_used)).filter(UsageLog.user_id == user_id).scalar() or 0
    
    # Get current month usage
    start_of_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
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
    request: Request,
    db: Session = Depends(get_db)
):
    """Get current user information"""
    current_user = await auth_handler.get_current_user(request, db)
    return current_user

@router.get("/users/me/usage", response_model=List[UsageLogResponse])
async def get_current_user_usage(
    request: Request,
    db: Session = Depends(get_db),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get current user's usage statistics"""
    current_user = await auth_handler.get_current_user(request, db)
    return await get_user_usage(current_user.id, start_date, end_date, request, db)

@router.get("/users/me/usage/summary")
async def get_current_user_usage_summary(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get current user's usage summary"""
    current_user = await auth_handler.get_current_user(request, db)
    return await get_user_usage_summary(current_user.id, request, db)

# --- Configuration Management Routes ---
@router.get("/config")
async def get_config(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get system configuration (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
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
    request: Request,
    db: Session = Depends(get_db)
):
    """Update system configuration (admin only)"""
    current_user = await auth_handler.get_current_admin_user(request, db)
    # In a real implementation, you would save these to a config file or database
    # For now, just return success
    return {"message": "Configuration updated successfully"}
