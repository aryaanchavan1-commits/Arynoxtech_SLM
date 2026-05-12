"""
Authentication & User Data Persistence for World Model SLM 2026 UI.
Supports username/password login, registration, per-user chat history,
and user data export.
"""
import os
import json
import hashlib
import secrets
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Try bcrypt, fallback to hashlib.pbkdf2
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

USERS_FILE = Path("./data/users.json")
USER_DATA_DIR = Path("./data/user_data")


def _ensure_dirs():
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _hash_password(password: str) -> str:
    """Hash a password securely."""
    if HAS_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()
    else:
        # Fallback PBKDF2
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return f"pbkdf2:{salt}:{hashed.hex()}"


def _verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    if HAS_BCRYPT:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    else:
        if not hashed.startswith("pbkdf2:"):
            return False
        _, salt, stored_hash = hashed.split(":")
        computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
        return secrets.compare_digest(computed.hex(), stored_hash)


def _load_users() -> Dict[str, Dict[str, Any]]:
    """Load users from JSON file."""
    _ensure_dirs()
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: Dict[str, Dict[str, Any]]) -> None:
    """Save users to JSON file."""
    _ensure_dirs()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def register_user(username: str, password: str, display_name: Optional[str] = None) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success, message).
    """
    if not username or not password:
        return False, "Username and password are required."

    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if username in users:
        return False, "Username already exists. Please log in or choose a different name."

    users[username] = {
        "username": username,
        "display_name": display_name or username.title(),
        "password_hash": _hash_password(password),
        "created_at": datetime.now().isoformat(),
        "last_login": None,
        "login_count": 0,
    }
    _save_users(users)

    # Create empty user data file
    save_user_data(username, {"messages": [], "uploaded_files": [], "settings": {}})

    return True, "Registration successful! Please log in."


def authenticate_user(username: str, password: str) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Authenticate a user.
    Returns (success, user_data_dict or None).
    """
    if not username or not password:
        return False, None

    username = username.strip().lower()
    users = _load_users()
    user = users.get(username)

    if not user:
        return False, None

    if not _verify_password(password, user["password_hash"]):
        return False, None

    # Update last login
    user["last_login"] = datetime.now().isoformat()
    user["login_count"] = user.get("login_count", 0) + 1
    users[username] = user
    _save_users(users)

    return True, {
        "username": user["username"],
        "display_name": user["display_name"],
        "created_at": user["created_at"],
        "last_login": user["last_login"],
        "login_count": user["login_count"],
    }


def user_exists(username: str) -> bool:
    """Check if a user exists."""
    users = _load_users()
    return username.strip().lower() in users


def get_user_data_path(username: str) -> Path:
    """Get the path to a user's data file."""
    safe_name = "".join(c for c in username if c.isalnum() or c in "_-").lower()
    return USER_DATA_DIR / f"{safe_name}.json"


def save_user_data(username: str, data: Dict[str, Any]) -> None:
    """Save user-specific data (chat history, files, settings)."""
    _ensure_dirs()
    path = get_user_data_path(username)
    data["_saved_at"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_user_data(username: str) -> Dict[str, Any]:
    """Load user-specific data."""
    path = get_user_data_path(username)
    if not path.exists():
        return {"messages": [], "uploaded_files": [], "settings": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"messages": [], "uploaded_files": [], "settings": {}}


def export_user_data(username: str) -> Optional[str]:
    """Export all user data as a JSON string."""
    users = _load_users()
    user = users.get(username.strip().lower())
    if not user:
        return None

    user_data = load_user_data(username)
    export = {
        "profile": {
            "username": user["username"],
            "display_name": user["display_name"],
            "created_at": user["created_at"],
            "last_login": user["last_login"],
            "login_count": user["login_count"],
        },
        "data": user_data,
        "exported_at": datetime.now().isoformat(),
    }
    return json.dumps(export, indent=2, ensure_ascii=False)


def delete_user_account(username: str, password: str) -> tuple[bool, str]:
    """Delete a user account and all associated data."""
    username = username.strip().lower()
    success, _ = authenticate_user(username, password)
    if not success:
        return False, "Invalid username or password."

    users = _load_users()
    if username in users:
        del users[username]
        _save_users(users)

    # Delete user data file
    path = get_user_data_path(username)
    if path.exists():
        path.unlink()

    return True, "Account deleted successfully."


def list_all_users() -> List[str]:
    """List all registered usernames (admin use)."""
    users = _load_users()
    return list(users.keys())

