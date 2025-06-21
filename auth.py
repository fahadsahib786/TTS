"""
Authentication and authorization system for VoiceAI TTS Server
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from database import User, UserSession, get_db
import os

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    import secrets
    SECRET_KEY = secrets.token_urlsafe(32)
    print("WARNING: JWT_SECRET_KEY not set in environment. Using generated key for this session.")
    print(f"For production, set JWT_SECRET_KEY environment variable to: {SECRET_KEY}")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300
REFRESH_TOKEN_EXPIRE_DAYS = 30

security = HTTPBearer()

class AuthHandler:
    def __init__(self):
        self.secret_key = SECRET_KEY
        self.algorithm = ALGORITHM

    def create_access_token(self, data: dict) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> dict:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Security(security),
        db: Session = Depends(get_db),
        request: Request = None
    ) -> User:
        """Get current authenticated user from token"""
        try:
            # Handle case where credentials might be None (from middleware)
            if credentials is None:
                # Try to get token from request headers (set by middleware)
                auth_header = request.headers.get("authorization", "") if request else ""
                if not auth_header.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="No authentication token provided")
                token = auth_header[7:]
            else:
                token = credentials.credentials
            
            # First try to validate as a session token
            payload = self.decode_token(token)
            user_id: int = payload.get("sub")
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            
            # Get user
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            
            if not user.is_active:
                raise HTTPException(status_code=403, detail="User account is disabled")
            
            if user.is_expired():
                raise HTTPException(status_code=403, detail="User account has expired")
            
            # Check if this is a valid session token
            session = (
                db.query(UserSession)
                .filter(
                    UserSession.user_id == user_id,
                    UserSession.session_token == token,
                    UserSession.is_active == True
                )
                .first()
            )
            
            if session:
                # Valid session found, update last used time
                if not session.is_expired():
                    session.last_used = datetime.utcnow()
                    if request:
                        session.ip_address = request.client.host
                        session.user_agent = request.headers.get("user-agent")
                    db.commit()
                    return user
                else:
                    # Session expired but token still valid, create new session
                    new_session = UserSession(
                        user_id=user_id,
                        session_token=token,
                        refresh_token=None,  # No refresh token for auto-renewed session
                        expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
                        ip_address=request.client.host if request else None,
                        user_agent=request.headers.get("user-agent") if request else None
                    )
                    db.add(new_session)
                    db.commit()
                    return user
            
            # No valid session found but token is valid, create new session
            new_session = UserSession(
                user_id=user_id,
                session_token=token,
                refresh_token=None,  # No refresh token for auto-renewed session
                expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
                ip_address=request.client.host if request else None,
                user_agent=request.headers.get("user-agent") if request else None
            )
            db.add(new_session)
            db.commit()
            return user
            
        except JWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_active_user(
        self,
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Get current active user"""
        if not current_user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user

    async def get_current_admin_user(
        self,
        current_user: User = Depends(get_current_user)
    ) -> User:
        """Get current admin user"""
        if not current_user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="Not enough permissions. Admin access required."
            )
        return current_user

auth_handler = AuthHandler()

async def create_user_session(
    user: User,
    db: Session,
    request: Request = None,
    response = None
) -> Dict[str, Any]:
    """Create new user session with tokens"""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[Session] Creating session for user {user.username} (ID: {user.id})")
    
    # Create tokens
    logger.info(f"[Session] Creating access token for user {user.id}")
    access_token = auth_handler.create_access_token({"sub": user.id})
    logger.info(f"[Session] Access token created: {access_token[:20]}...")
    
    logger.info(f"[Session] Creating refresh token for user {user.id}")
    refresh_token = auth_handler.create_refresh_token({"sub": user.id})
    logger.info(f"[Session] Refresh token created: {refresh_token[:20]}...")
    
    # Create session record
    session_expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    logger.info(f"[Session] Session will expire at: {session_expires_at}")
    
    session = UserSession(
        user_id=user.id,
        session_token=access_token,
        refresh_token=refresh_token,
        expires_at=session_expires_at
    )
    
    if request:
        session.ip_address = request.client.host
        session.user_agent = request.headers.get("user-agent")
        logger.info(f"[Session] Session IP: {session.ip_address}")
        logger.info(f"[Session] Session User-Agent: {session.user_agent}")
    
    logger.info(f"[Session] Adding session to database")
    db.add(session)
    db.commit()
    logger.info(f"[Session] Session saved to database with ID: {session.id}")
    
    # Set secure cookies if response object is provided
    if response:
        logger.info(f"[Session] Setting HTTP-only cookies in response")
        # Determine if we're in a secure environment
        is_secure = request.url.scheme == "https" if request else False
        logger.info(f"[Session] Cookie secure flag: {is_secure}")
        
        access_cookie_max_age = ACCESS_TOKEN_EXPIRE_MINUTES * 60
        refresh_cookie_max_age = REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        
        logger.info(f"[Session] Access cookie max age: {access_cookie_max_age} seconds")
        logger.info(f"[Session] Refresh cookie max age: {refresh_cookie_max_age} seconds")
        
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=access_cookie_max_age,
            httponly=True,
            secure=is_secure,  # Only set secure flag if using HTTPS
            samesite="lax"  # Changed to lax for better compatibility
        )
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            max_age=refresh_cookie_max_age,
            httponly=True,
            secure=is_secure,
            samesite="lax"
        )
        logger.info(f"[Session] HTTP-only cookies set in response")
    else:
        logger.warning(f"[Session] No response object provided, cookies not set")
    
    user_data = {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "is_admin": user.is_admin,
        "monthly_char_limit": user.monthly_char_limit,
        "daily_char_limit": user.daily_char_limit,
        "per_request_char_limit": user.per_request_char_limit,
        "chars_used_current_month": user.chars_used_current_month,
        "chars_used_today": user.chars_used_today
    }
    
    result = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": user_data
    }
    
    logger.info(f"[Session] Session creation completed for user {user.username}")
    logger.info(f"[Session] Returning session data with token type: {result['token_type']}")
    
    return result

async def refresh_access_token(
    refresh_token: str,
    db: Session,
    request: Request = None
) -> Dict[str, Any]:
    """Create new access token using refresh token"""
    try:
        # Validate refresh token
        payload = auth_handler.decode_token(refresh_token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Get session
        session = (
            db.query(UserSession)
            .filter(
                UserSession.user_id == user_id,
                UserSession.refresh_token == refresh_token,
                UserSession.is_active == True
            )
            .first()
        )
        
        if not session or session.is_expired():
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
        
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active or user.is_expired():
            raise HTTPException(status_code=401, detail="User is inactive or expired")
        
        # Create new access token
        new_access_token = auth_handler.create_access_token({"sub": user_id})
        
        # Update session
        session.session_token = new_access_token
        session.last_used = datetime.utcnow()
        if request:
            session.ip_address = request.client.host
            session.user_agent = request.headers.get("user-agent")
        
        db.commit()
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

async def logout_user(
    user: User,
    db: Session,
    current_token: str = None,
    all_sessions: bool = False
) -> None:
    """Logout user by invalidating sessions"""
    query = db.query(UserSession).filter(UserSession.user_id == user.id)
    
    if not all_sessions and current_token:
        # Only invalidate current session
        query = query.filter(UserSession.session_token == current_token)
    
    # Mark sessions as inactive
    query.update({"is_active": False})
    db.commit()
