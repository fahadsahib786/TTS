#!/usr/bin/env python3
"""
Seed script to populate the database with users
"""
import sys
import os
import random
import string
from datetime import datetime
from sqlalchemy.orm import Session

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_db, User, init_database

def generate_secure_password(length=16):
    """Generate a secure random password"""
    # Define character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = "!@#$%^&*"
    
    # Ensure at least one character from each set
    password = [
        random.choice(lowercase),
        random.choice(uppercase),
        random.choice(digits),
        random.choice(special)
    ]
    
    # Fill remaining length with random characters from all sets
    all_chars = lowercase + uppercase + digits + special
    password.extend(random.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password
    random.shuffle(password)
    return ''.join(password)

def generate_random_email():
    """Generate a random email address"""
    # Define character sets and domains
    chars = string.ascii_lowercase + string.digits
    domains = ["gmail.com", "yahoo.com", "outlook.com", "protonmail.com", "icloud.com"]
    
    # Generate random username (8-12 characters)
    username_length = random.randint(8, 12)
    username = ''.join(random.choice(chars) for _ in range(username_length))
    
    # Select random domain
    domain = random.choice(domains)
    
    return f"{username}@{domain}"

def create_users():
    """Create users with specified limits"""
    db = next(get_db())
    users_created = []
    
    try:
        # Create admin user with no limits
        admin_email = "admin@voiceai.local"
        admin_pass = generate_secure_password()
        
        admin = User(
            email=admin_email,
            username="admin",
            hashed_password=User.hash_password(admin_pass),
            full_name="System Administrator",
            is_admin=True,
            is_active=True,
            monthly_char_limit=-1,  # No limit
            daily_char_limit=-1,    # No limit
            per_request_char_limit=-1,  # No limit
            expires_at=None  # Never expires
        )
        db.add(admin)
        users_created.append({
            "type": "Admin",
            "email": admin_email,
            "password": admin_pass,
            "limit": "Unlimited"
        })

        # Create 10 users with 10k monthly limit
        for _ in range(10):
            email = generate_random_email()
            password = generate_secure_password()
            
            user = User(
                email=email,
                username=email.split('@')[0],
                hashed_password=User.hash_password(password),
                full_name="Standard User",
                is_admin=False,
                is_active=True,
                monthly_char_limit=10000,   # 10k monthly
                daily_char_limit=2000,      # 2k daily
                per_request_char_limit=1000  # 1k per request
            )
            db.add(user)
            users_created.append({
                "type": "Standard",
                "email": email,
                "password": password,
                "limit": "10k monthly"
            })

        # Create 15 users with 20M monthly limit
        for _ in range(15):
            email = generate_random_email()
            password = generate_secure_password()
            
            user = User(
                email=email,
                username=email.split('@')[0],
                hashed_password=User.hash_password(password),
                full_name="Premium User",
                is_admin=False,
                is_active=True,
                monthly_char_limit=20000000,  # 20M monthly
                daily_char_limit=500000,      # 500k daily
                per_request_char_limit=30000  # 30k per request
            )
            db.add(user)
            users_created.append({
                "type": "Premium",
                "email": email,
                "password": password,
                "limit": "20M monthly"
            })

        # Commit all changes
        db.commit()

        # Save credentials to a file
        with open('user_credentials.txt', 'w') as f:
            f.write("VoiceAI TTS Server - User Credentials\n")
            f.write("=" * 50 + "\n\n")
            
            # Write admin credentials first
            admin_cred = next(cred for cred in users_created if cred["type"] == "Admin")
            f.write("ADMIN USER:\n")
            f.write(f"Email: {admin_cred['email']}\n")
            f.write(f"Password: {admin_cred['password']}\n")
            f.write("Limit: Unlimited\n\n")
            
            f.write("STANDARD USERS (10k monthly limit):\n")
            f.write("-" * 50 + "\n")
            for cred in users_created:
                if cred["type"] == "Standard":
                    f.write(f"Email: {cred['email']}\n")
                    f.write(f"Password: {cred['password']}\n")
                    f.write("-" * 50 + "\n")
            
            f.write("\nPREMIUM USERS (20M monthly limit):\n")
            f.write("-" * 50 + "\n")
            for cred in users_created:
                if cred["type"] == "Premium":
                    f.write(f"Email: {cred['email']}\n")
                    f.write(f"Password: {cred['password']}\n")
                    f.write("-" * 50 + "\n")

        print(f"\n✅ Successfully created {len(users_created)} users!")
        print("\nUser Summary:")
        print("- 1 admin user (unlimited)")
        print("- 10 standard users (10k monthly, 2k daily, 1k per request)")
        print("- 15 premium users (20M monthly, 500k daily, 30k per request)")
        print("\n📝 User credentials have been saved to 'user_credentials.txt'")

    except Exception as e:
        print(f"❌ Error creating users: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    print("🚀 VoiceAI TTS Server - Database Seeder")
    print("=" * 50)
    
    print("This will create users in your database.")
    print("Press Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
        # Initialize database tables
        init_database()
        # Create users
        create_users()
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user.")
        sys.exit(1)
