"""
Database models and configuration for VoiceAI TTS Server
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime, timedelta
import hashlib
import secrets
import os
import bcrypt

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    monthly_char_limit = Column(Integer, default=10000)  # Default 10k characters per month
    daily_char_limit = Column(Integer, default=1000)    # Default 1k characters per day
    per_request_char_limit = Column(Integer, default=500)  # Default 500 characters per request
    chars_used_current_month = Column(Integer, default=0)
    chars_used_today = Column(Integer, default=0)
    last_reset_date = Column(DateTime, default=func.now())
    last_daily_reset = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime, nullable=True)  # Account expiry date
    
    # Relationships
    usage_logs = relationship("UsageLog", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    
    def check_password(self, password: str) -> bool:
        """Check if provided password matches the hashed password"""
        try:
            return bcrypt.checkpw(password.encode(), self.hashed_password.encode())
        except Exception:
            return False
    
    @staticmethod
    def validate_password_complexity(password: str) -> tuple[bool, str]:
        """Validate password complexity requirements"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            return False, "Password must contain at least one special character"
        
        return True, "Password meets complexity requirements"
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def reset_monthly_usage(self):
        """Reset monthly usage if a new month has started"""
        now = datetime.utcnow()
        if self.last_reset_date.month != now.month or self.last_reset_date.year != now.year:
            self.chars_used_current_month = 0
            self.last_reset_date = now
    
    def reset_daily_usage(self):
        """Reset daily usage if a new day has started"""
        now = datetime.utcnow()
        if self.last_daily_reset.date() != now.date():
            self.chars_used_today = 0
            self.last_daily_reset = now
    
    def can_use_characters(self, char_count: int) -> tuple[bool, str]:
        """Check if user can use the specified number of characters"""
        if self.is_admin:
            return True, "Admin user - unlimited access"
        
        # Check per-request limit
        if char_count > self.per_request_char_limit:
            return False, f"Request exceeds per-request limit of {self.per_request_char_limit} characters"
        
        # Reset counters if needed
        self.reset_monthly_usage()
        self.reset_daily_usage()
        
        # Check monthly limit
        if (self.chars_used_current_month + char_count) > self.monthly_char_limit:
            return False, f"Monthly character limit of {self.monthly_char_limit} exceeded"
        
        # Check daily limit
        if (self.chars_used_today + char_count) > self.daily_char_limit:
            return False, f"Daily character limit of {self.daily_char_limit} exceeded"
        
        return True, "Character usage allowed"
    
    def use_characters(self, char_count: int) -> bool:
        """Use characters from user's quota"""
        if self.is_admin:
            return True
            
        can_use, _ = self.can_use_characters(char_count)
        if can_use:
            self.chars_used_current_month += char_count
            self.chars_used_today += char_count
            return True
        return False
    
    def is_expired(self) -> bool:
        """Check if user account has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

class UsageLog(Base):
    __tablename__ = "usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    characters_used = Column(Integer, nullable=False)
    text_content = Column(Text, nullable=True)  # Store first 500 chars for reference
    voice_mode = Column(String(50), nullable=True)
    voice_file = Column(String(255), nullable=True)
    generation_time = Column(Float, nullable=True)  # Time taken to generate
    created_at = Column(DateTime, default=func.now())
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="usage_logs")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
    last_used = Column(DateTime, default=func.now())
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.utcnow() > self.expires_at
    
    @staticmethod
    def generate_tokens():
        """Generate session and refresh tokens"""
        session_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)
        return session_token, refresh_token

class GenerationQueue(Base):
    __tablename__ = "generation_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    text_content = Column(Text, nullable=False)
    parameters = Column(Text, nullable=True)  # JSON string of generation parameters
    result_file_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    priority = Column(Integer, default=0)  # Higher number = higher priority

class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), nullable=False, index=True)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(String(500), nullable=True)
    success = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    
    @staticmethod
    def is_ip_blocked(ip_address: str, db_session) -> bool:
        """Check if IP is blocked due to too many failed attempts"""
        # Block if more than 5 failed attempts in last 15 minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=15)
        failed_attempts = db_session.query(LoginAttempt).filter(
            LoginAttempt.ip_address == ip_address,
            LoginAttempt.success == False,
            LoginAttempt.created_at >= cutoff_time
        ).count()
        return failed_attempts >= 5
    
    @staticmethod
    def is_email_blocked(email: str, db_session) -> bool:
        """Check if email is blocked due to too many failed attempts"""
        # Block if more than 10 failed attempts in last 30 minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        failed_attempts = db_session.query(LoginAttempt).filter(
            LoginAttempt.email == email,
            LoginAttempt.success == False,
            LoginAttempt.created_at >= cutoff_time
        ).count()
        return failed_attempts >= 10

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./voiceai.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_database():
    """Initialize database tables only"""
    create_tables()
    print("Database tables created successfully")

if __name__ == "__main__":
    init_database()
