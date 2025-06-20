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
    monthly_char_limit = Column(Integer, default=10000)  # Default 10k characters
    chars_used_current_month = Column(Integer, default=0)
    last_reset_date = Column(DateTime, default=func.now())
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime, nullable=True)  # Account expiry date
    
    # Relationships
    usage_logs = relationship("UsageLog", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    
    def check_password(self, password: str) -> bool:
        """Check if provided password matches the hashed password"""
        return self.hashed_password == self.hash_password(password)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def reset_monthly_usage(self):
        """Reset monthly usage if a new month has started"""
        now = datetime.utcnow()
        if self.last_reset_date.month != now.month or self.last_reset_date.year != now.year:
            self.chars_used_current_month = 0
            self.last_reset_date = now
    
    def can_use_characters(self, char_count: int) -> bool:
        """Check if user can use the specified number of characters"""
        if self.is_admin:
            return True
        
        self.reset_monthly_usage()
        return (self.chars_used_current_month + char_count) <= self.monthly_char_limit
    
    def use_characters(self, char_count: int) -> bool:
        """Use characters from user's quota"""
        if self.is_admin:
            return True
            
        if self.can_use_characters(char_count):
            self.chars_used_current_month += char_count
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
    """Initialize database with default data"""
    create_tables()
    
    # Create default admin user if not exists
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.email == "admin@voiceai.com").first()
        if not admin_user:
            admin_user = User(
                email="admin@voiceai.com",
                username="admin",
                hashed_password=User.hash_password("admin123"),
                full_name="VoiceAI Administrator",
                is_admin=True,
                is_active=True,
                monthly_char_limit=0,  # Unlimited for admin
                expires_at=None  # Never expires
            )
            db.add(admin_user)
            
            # Create 20 test users
            test_users = []
            
            # Create 20 test users (10 premium, 10 trial)
            for i in range(1, 21):
                is_premium = i <= 10
                user = User(
                    email=f"{'premium' if is_premium else 'trial'}{i if is_premium else i-10}@voiceai.com",
                    username=f"{'premium' if is_premium else 'trial'}{i if is_premium else i-10}",
                    hashed_password=User.hash_password("password123"),
                    full_name=f"{'Premium' if is_premium else 'Trial'} User {i if is_premium else i-10}",
                    is_admin=False,
                    is_active=True,
                    monthly_char_limit=20000000 if is_premium else 10000,  # 20M for premium, 10k for trial
                    expires_at=datetime.utcnow() + timedelta(days=30)  # 30 days expiry for all users
                )
                test_users.append(user)
            
            db.add_all(test_users)
            db.commit()
            print("Database initialized with default users")
            print("Admin: admin@voiceai.com / admin123 (never expires)")
            print("Premium users: premium1@voiceai.com to premium10@voiceai.com / password123 (20M chars, 30 days expiry)")
            print("Trial users: trial1@voiceai.com to trial10@voiceai.com / password123 (10k chars, 30 days expiry)")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
