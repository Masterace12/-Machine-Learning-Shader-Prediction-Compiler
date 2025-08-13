#!/usr/bin/env python3
"""
Access Control System with Reputation-Based Permissions

This module implements a comprehensive access control system for the shader 
distribution network, featuring:

- Multi-tier permission models (Anonymous, Verified, Trusted, Admin)
- Reputation-based access control with dynamic scoring
- Rate limiting and quota management
- Role-based permissions for different operations
- Security quarantine system for suspicious users
- JWT-based authentication for API access
- Gradual trust building through successful validations
- Abuse detection and mitigation

The system ensures that shader sharing is secure while enabling legitimate
users to participate effectively in the community.
"""

import jwt
import time
import json
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from pathlib import Path
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
import sqlite3
import contextlib

# Try to import bcrypt for password hashing
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles in the system"""
    ANONYMOUS = "anonymous"      # Unverified users
    VERIFIED = "verified"        # Email/Steam verified users
    TRUSTED = "trusted"          # High reputation users
    MODERATOR = "moderator"      # Community moderators  
    ADMIN = "admin"             # System administrators
    BANNED = "banned"           # Banned users


class Permission(Enum):
    """System permissions"""
    # Basic operations
    VIEW_SHADERS = "view_shaders"
    DOWNLOAD_SHADERS = "download_shaders"
    UPLOAD_SHADERS = "upload_shaders"
    RATE_SHADERS = "rate_shaders"
    COMMENT_SHADERS = "comment_shaders"
    
    # Advanced operations
    MODERATE_CONTENT = "moderate_content"
    BAN_USERS = "ban_users"
    ADMIN_PANEL = "admin_panel"
    SYSTEM_CONFIG = "system_config"
    VIEW_ANALYTICS = "view_analytics"
    
    # API access
    API_READ = "api_read"
    API_WRITE = "api_write"
    API_ADMIN = "api_admin"


class ReputationAction(Enum):
    """Actions that affect reputation"""
    SHADER_UPLOAD = "shader_upload"
    SHADER_DOWNLOAD = "shader_download"
    SUCCESSFUL_VALIDATION = "successful_validation"
    FAILED_VALIDATION = "failed_validation"
    POSITIVE_RATING = "positive_rating"
    NEGATIVE_RATING = "negative_rating"
    REPORT_MALICIOUS = "report_malicious"
    CONFIRMED_MALICIOUS = "confirmed_malicious"
    HELPFUL_COMMENT = "helpful_comment"
    SPAM_DETECTED = "spam_detected"
    ABUSE_REPORTED = "abuse_reported"


class QuarantineReason(Enum):
    """Reasons for user quarantine"""
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALICIOUS_UPLOADS = "malicious_uploads"
    SPAM_BEHAVIOR = "spam_behavior"
    ABUSE_REPORTS = "abuse_reports"
    RATE_LIMIT_VIOLATIONS = "rate_limit_violations"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class ReputationScore:
    """User reputation scoring"""
    user_id: str
    current_score: float
    max_score: float
    min_score: float
    
    # Score breakdown
    upload_score: float = 0.0
    validation_score: float = 0.0
    community_score: float = 0.0
    security_score: float = 100.0
    
    # Metadata
    last_updated: float = field(default_factory=time.time)
    score_history: List[Tuple[float, str, float]] = field(default_factory=list)  # (timestamp, action, delta)
    
    def add_score_change(self, action: ReputationAction, delta: float, reason: str = ""):
        """Add a score change to history"""
        current_time = time.time()
        self.score_history.append((current_time, f"{action.value}:{reason}", delta))
        
        # Keep only recent history
        if len(self.score_history) > 1000:
            self.score_history = self.score_history[-500:]
        
        self.last_updated = current_time
    
    def get_recent_activity_score(self, hours: int = 24) -> float:
        """Get reputation change in recent hours"""
        cutoff = time.time() - (hours * 3600)
        recent_changes = [delta for timestamp, _, delta in self.score_history if timestamp > cutoff]
        return sum(recent_changes)


@dataclass
class RateLimitRule:
    """Rate limiting rule"""
    name: str
    max_requests: int
    window_seconds: int
    burst_allowance: int = 0  # Additional requests allowed in burst
    reset_policy: str = "sliding"  # "sliding" or "fixed"


@dataclass
class UserSession:
    """User session information"""
    session_id: str
    user_id: str
    role: UserRole
    permissions: Set[Permission]
    created_at: float
    last_activity: float
    ip_address_hash: str
    user_agent_hash: str
    
    # Session limits
    rate_limits: Dict[str, Tuple[int, float]] = field(default_factory=dict)  # (count, window_start)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def check_rate_limit(self, rule: RateLimitRule) -> bool:
        """Check if action is within rate limit"""
        current_time = time.time()
        
        if rule.name not in self.rate_limits:
            self.rate_limits[rule.name] = (0, current_time)
            return True
        
        count, window_start = self.rate_limits[rule.name]
        
        if rule.reset_policy == "fixed":
            # Fixed window
            if current_time - window_start >= rule.window_seconds:
                self.rate_limits[rule.name] = (0, current_time)
                return True
        else:
            # Sliding window
            if current_time - window_start >= rule.window_seconds:
                # Reset sliding window
                self.rate_limits[rule.name] = (0, current_time)
                return True
        
        # Check if within limit
        max_allowed = rule.max_requests + rule.burst_allowance
        return count < max_allowed
    
    def increment_rate_limit(self, rule_name: str):
        """Increment rate limit counter"""
        if rule_name in self.rate_limits:
            count, window_start = self.rate_limits[rule_name]
            self.rate_limits[rule_name] = (count + 1, window_start)
        else:
            self.rate_limits[rule_name] = (1, time.time())


@dataclass
class QuarantineRecord:
    """User quarantine record"""
    user_id: str
    reason: QuarantineReason
    start_time: float
    end_time: Optional[float]
    severity: int  # 1-10 scale
    appeal_allowed: bool
    quarantine_restrictions: Set[str]
    evidence: Dict[str, Any]
    
    def is_active(self) -> bool:
        """Check if quarantine is still active"""
        current_time = time.time()
        if self.end_time is None:
            return True  # Permanent quarantine
        return current_time < self.end_time


class ReputationEngine:
    """Calculate and manage user reputation scores"""
    
    def __init__(self):
        # Reputation scoring weights
        self.action_weights = {
            ReputationAction.SHADER_UPLOAD: 2.0,
            ReputationAction.SUCCESSFUL_VALIDATION: 5.0,
            ReputationAction.FAILED_VALIDATION: -10.0,
            ReputationAction.POSITIVE_RATING: 1.0,
            ReputationAction.NEGATIVE_RATING: 0.5,
            ReputationAction.REPORT_MALICIOUS: 3.0,
            ReputationAction.CONFIRMED_MALICIOUS: -50.0,
            ReputationAction.HELPFUL_COMMENT: 1.0,
            ReputationAction.SPAM_DETECTED: -20.0,
            ReputationAction.ABUSE_REPORTED: -15.0
        }
        
        # Score boundaries for role promotion
        self.role_thresholds = {
            UserRole.ANONYMOUS: (0, 10),
            UserRole.VERIFIED: (10, 50),
            UserRole.TRUSTED: (50, 200),
            UserRole.MODERATOR: (200, float('inf'))
        }
        
        # Decay rates (reputation slowly decreases over time if inactive)
        self.decay_rate = 0.1  # Points lost per day of inactivity
        self.max_decay_days = 30  # Maximum decay period
    
    def calculate_base_score(self, actions: List[Tuple[ReputationAction, float, str]]) -> float:
        """Calculate base reputation score from actions"""
        score = 0.0
        
        for action, timestamp, reason in actions:
            weight = self.action_weights.get(action, 0.0)
            
            # Apply time-based weighting (recent actions matter more)
            age_days = (time.time() - timestamp) / (24 * 3600)
            time_weight = max(0.1, 1.0 - (age_days / 365))  # Decay over a year
            
            score += weight * time_weight
        
        return max(0.0, score)
    
    def calculate_security_score(self, user_id: str, security_events: List[Dict]) -> float:
        """Calculate security reputation score"""
        base_security = 100.0
        
        for event in security_events:
            event_type = event.get('type', '')
            severity = event.get('severity', 1)
            
            if event_type == 'malicious_upload':
                base_security -= severity * 20
            elif event_type == 'suspicious_activity':
                base_security -= severity * 5
            elif event_type == 'rate_limit_violation':
                base_security -= severity * 2
            elif event_type == 'security_validation_passed':
                base_security += 1
        
        return max(0.0, min(100.0, base_security))
    
    def calculate_community_score(self, ratings: List[int], comments: List[Dict]) -> float:
        """Calculate community interaction score"""
        score = 0.0
        
        # Rating contribution
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            score += (avg_rating - 3.0) * 2.0  # Center around 3.0 rating
        
        # Comment contribution
        helpful_comments = sum(1 for c in comments if c.get('helpful_votes', 0) > 0)
        score += helpful_comments * 0.5
        
        return max(0.0, score)
    
    def apply_reputation_decay(self, current_score: float, last_activity: float) -> float:
        """Apply time-based reputation decay"""
        days_inactive = (time.time() - last_activity) / (24 * 3600)
        
        if days_inactive > 1.0:
            decay_days = min(days_inactive, self.max_decay_days)
            decay_amount = decay_days * self.decay_rate
            return max(0.0, current_score - decay_amount)
        
        return current_score
    
    def suggest_role_for_score(self, score: float) -> UserRole:
        """Suggest user role based on reputation score"""
        for role, (min_score, max_score) in self.role_thresholds.items():
            if min_score <= score < max_score:
                return role
        
        return UserRole.ANONYMOUS


class AccessControlDatabase:
    """SQLite database for access control data"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("access_control.db")
        self.lock = threading.Lock()
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email_hash TEXT,
                    password_hash TEXT,
                    role TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_activity REAL NOT NULL,
                    email_verified BOOLEAN DEFAULT FALSE,
                    steam_verified BOOLEAN DEFAULT FALSE,
                    reputation_score REAL DEFAULT 0.0,
                    security_score REAL DEFAULT 100.0,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS reputation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    score_delta REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    reason TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                );
                
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_activity REAL NOT NULL,
                    ip_address_hash TEXT,
                    user_agent_hash TEXT,
                    rate_limits TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                );
                
                CREATE TABLE IF NOT EXISTS quarantine_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    severity INTEGER NOT NULL,
                    evidence TEXT,
                    active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                );
                
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    details TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
                CREATE INDEX IF NOT EXISTS idx_reputation_user_time ON reputation_history(user_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_quarantine_active ON quarantine_records(active, user_id);
                CREATE INDEX IF NOT EXISTS idx_security_events_user ON security_events(user_id, timestamp);
            """)
    
    @contextlib.contextmanager
    def get_connection(self):
        """Get database connection with proper locking"""
        with self.lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def create_user(self, user_id: str, email: str = None, password: str = None,
                   role: UserRole = UserRole.ANONYMOUS) -> bool:
        """Create a new user"""
        try:
            email_hash = None
            password_hash = None
            
            if email:
                email_hash = hashlib.sha256(email.lower().encode()).hexdigest()
            
            if password and HAS_BCRYPT:
                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            
            current_time = time.time()
            
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO users (user_id, email_hash, password_hash, role, 
                                     created_at, last_activity, reputation_score, security_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (user_id, email_hash, password_hash, role.value, 
                     current_time, current_time, 0.0, 100.0))
                conn.commit()
            
            logger.info(f"Created user {user_id} with role {role.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating user {user_id}: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information"""
        try:
            with self.get_connection() as conn:
                row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
                if row:
                    return dict(row)
            return None
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None
    
    def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp"""
        try:
            with self.get_connection() as conn:
                conn.execute("UPDATE users SET last_activity = ? WHERE user_id = ?",
                           (time.time(), user_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating activity for {user_id}: {e}")
    
    def add_reputation_change(self, user_id: str, action: ReputationAction, 
                            score_delta: float, reason: str = ""):
        """Add reputation change record"""
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO reputation_history (user_id, action, score_delta, timestamp, reason)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, action.value, score_delta, time.time(), reason))
                
                # Update user's current reputation score
                conn.execute("""
                    UPDATE users SET reputation_score = reputation_score + ?
                    WHERE user_id = ?
                """, (score_delta, user_id))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error adding reputation change for {user_id}: {e}")
    
    def get_reputation_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get user's reputation history"""
        try:
            cutoff = time.time() - (days * 24 * 3600)
            with self.get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM reputation_history 
                    WHERE user_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (user_id, cutoff)).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting reputation history for {user_id}: {e}")
            return []


class AccessControlSystem:
    """Main access control system"""
    
    def __init__(self, db_path: Path = None, jwt_secret: str = None):
        self.db = AccessControlDatabase(db_path)
        self.reputation_engine = ReputationEngine()
        
        # JWT configuration
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_hours = 24
        
        # Active sessions
        self.active_sessions: Dict[str, UserSession] = {}
        self.session_lock = threading.Lock()
        
        # Rate limiting rules
        self.rate_limit_rules = {
            "shader_upload": RateLimitRule("shader_upload", 10, 3600),  # 10 per hour
            "shader_download": RateLimitRule("shader_download", 100, 3600),  # 100 per hour
            "api_request": RateLimitRule("api_request", 1000, 3600),  # 1000 per hour
            "rating_submission": RateLimitRule("rating_submission", 20, 3600),  # 20 per hour
        }
        
        # Role permissions
        self.role_permissions = {
            UserRole.ANONYMOUS: {
                Permission.VIEW_SHADERS, Permission.DOWNLOAD_SHADERS
            },
            UserRole.VERIFIED: {
                Permission.VIEW_SHADERS, Permission.DOWNLOAD_SHADERS,
                Permission.UPLOAD_SHADERS, Permission.RATE_SHADERS,
                Permission.API_READ
            },
            UserRole.TRUSTED: {
                Permission.VIEW_SHADERS, Permission.DOWNLOAD_SHADERS,
                Permission.UPLOAD_SHADERS, Permission.RATE_SHADERS,
                Permission.COMMENT_SHADERS, Permission.API_READ, Permission.API_WRITE
            },
            UserRole.MODERATOR: {
                Permission.VIEW_SHADERS, Permission.DOWNLOAD_SHADERS,
                Permission.UPLOAD_SHADERS, Permission.RATE_SHADERS,
                Permission.COMMENT_SHADERS, Permission.MODERATE_CONTENT,
                Permission.API_READ, Permission.API_WRITE, Permission.VIEW_ANALYTICS
            },
            UserRole.ADMIN: set(Permission),  # All permissions
            UserRole.BANNED: set()  # No permissions
        }
        
        logger.info("Access control system initialized")
    
    def create_user(self, user_id: str, email: str = None, password: str = None,
                   initial_role: UserRole = UserRole.ANONYMOUS) -> bool:
        """Create a new user account"""
        return self.db.create_user(user_id, email, password, initial_role)
    
    def authenticate_user(self, user_id: str, password: str = None) -> Optional[str]:
        """Authenticate user and return JWT token"""
        user_data = self.db.get_user(user_id)
        if not user_data:
            return None
        
        # Check password if provided
        if password and user_data.get('password_hash'):
            if not HAS_BCRYPT:
                logger.warning("bcrypt not available, password authentication disabled")
                return None
            
            if not bcrypt.checkpw(password.encode(), user_data['password_hash'].encode()):
                return None
        
        # Check if user is banned
        if UserRole(user_data['role']) == UserRole.BANNED:
            logger.warning(f"Banned user {user_id} attempted to authenticate")
            return None
        
        # Generate JWT token
        token_data = {
            'user_id': user_id,
            'role': user_data['role'],
            'iat': time.time(),
            'exp': time.time() + (self.jwt_expiry_hours * 3600)
        }
        
        token = jwt.encode(token_data, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        # Update user activity
        self.db.update_user_activity(user_id)
        
        logger.info(f"User {user_id} authenticated successfully")
        return token
    
    def create_session(self, token: str, ip_address: str = None, 
                      user_agent: str = None) -> Optional[UserSession]:
        """Create user session from JWT token"""
        try:
            # Decode JWT token
            token_data = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            user_id = token_data['user_id']
            role = UserRole(token_data['role'])
            
            # Get user permissions
            permissions = self.role_permissions.get(role, set())
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            
            # Hash IP and user agent for privacy
            ip_hash = hashlib.sha256((ip_address or "").encode()).hexdigest()
            ua_hash = hashlib.sha256((user_agent or "").encode()).hexdigest()
            
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                role=role,
                permissions=permissions,
                created_at=time.time(),
                last_activity=time.time(),
                ip_address_hash=ip_hash,
                user_agent_hash=ua_hash
            )
            
            # Store session
            with self.session_lock:
                self.active_sessions[session_id] = session
            
            logger.info(f"Created session {session_id} for user {user_id}")
            return session
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token provided")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get active session"""
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.update_activity()
            return session
    
    def check_permission(self, session_id: str, permission: Permission) -> bool:
        """Check if session has specific permission"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        return permission in session.permissions
    
    def check_rate_limit(self, session_id: str, operation: str) -> bool:
        """Check if operation is within rate limits"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        rule = self.rate_limit_rules.get(operation)
        if not rule:
            return True
        
        # Higher reputation users get higher limits
        user_data = self.db.get_user(session.user_id)
        if user_data:
            reputation_multiplier = min(2.0, 1.0 + (user_data['reputation_score'] / 100.0))
            effective_rule = RateLimitRule(
                rule.name,
                int(rule.max_requests * reputation_multiplier),
                rule.window_seconds,
                rule.burst_allowance
            )
        else:
            effective_rule = rule
        
        if session.check_rate_limit(effective_rule):
            session.increment_rate_limit(operation)
            return True
        
        logger.warning(f"Rate limit exceeded for user {session.user_id}, operation {operation}")
        return False
    
    def update_reputation(self, user_id: str, action: ReputationAction, 
                         reason: str = "", multiplier: float = 1.0):
        """Update user reputation based on action"""
        base_weight = self.reputation_engine.action_weights.get(action, 0.0)
        score_delta = base_weight * multiplier
        
        self.db.add_reputation_change(user_id, action, score_delta, reason)
        
        # Check for role promotion
        user_data = self.db.get_user(user_id)
        if user_data:
            new_score = user_data['reputation_score'] + score_delta
            suggested_role = self.reputation_engine.suggest_role_for_score(new_score)
            
            current_role = UserRole(user_data['role'])
            if suggested_role != current_role and suggested_role != UserRole.BANNED:
                self._promote_user(user_id, suggested_role)
        
        logger.info(f"Updated reputation for {user_id}: {action.value} ({score_delta:+.1f})")
    
    def quarantine_user(self, user_id: str, reason: QuarantineReason, 
                       severity: int, duration_hours: Optional[int] = None,
                       evidence: Dict[str, Any] = None) -> bool:
        """Place user in quarantine"""
        try:
            current_time = time.time()
            end_time = None
            
            if duration_hours is not None:
                end_time = current_time + (duration_hours * 3600)
            
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO quarantine_records 
                    (user_id, reason, start_time, end_time, severity, evidence, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, reason.value, current_time, end_time, severity,
                     json.dumps(evidence or {}), True))
                
                # Suspend user temporarily
                if severity >= 7:  # High severity = temporary ban
                    conn.execute("UPDATE users SET role = ? WHERE user_id = ?",
                               (UserRole.BANNED.value, user_id))
                
                conn.commit()
            
            # Terminate active sessions for quarantined user
            self._terminate_user_sessions(user_id)
            
            logger.warning(f"User {user_id} quarantined: {reason.value} (severity {severity})")
            return True
            
        except Exception as e:
            logger.error(f"Error quarantining user {user_id}: {e}")
            return False
    
    def check_user_quarantine(self, user_id: str) -> Optional[QuarantineRecord]:
        """Check if user is currently quarantined"""
        try:
            with self.db.get_connection() as conn:
                row = conn.execute("""
                    SELECT * FROM quarantine_records 
                    WHERE user_id = ? AND active = TRUE
                    ORDER BY start_time DESC LIMIT 1
                """, (user_id,)).fetchone()
                
                if row:
                    record = QuarantineRecord(
                        user_id=row['user_id'],
                        reason=QuarantineReason(row['reason']),
                        start_time=row['start_time'],
                        end_time=row['end_time'],
                        severity=row['severity'],
                        appeal_allowed=True,  # Could be configurable
                        quarantine_restrictions=set(),
                        evidence=json.loads(row['evidence']) if row['evidence'] else {}
                    )
                    
                    if record.is_active():
                        return record
                    else:
                        # Quarantine expired, deactivate it
                        conn.execute("""
                            UPDATE quarantine_records SET active = FALSE 
                            WHERE id = ?
                        """, (row['id'],))
                        conn.commit()
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking quarantine for {user_id}: {e}")
            return None
    
    def record_security_event(self, user_id: str, event_type: str, 
                            severity: int, details: Dict[str, Any] = None):
        """Record security-related event"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO security_events (user_id, event_type, severity, timestamp, details)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, event_type, severity, time.time(), json.dumps(details or {})))
                conn.commit()
            
            # Check if user should be quarantined based on security events
            self._check_auto_quarantine(user_id)
            
        except Exception as e:
            logger.error(f"Error recording security event for {user_id}: {e}")
    
    def _promote_user(self, user_id: str, new_role: UserRole):
        """Promote user to new role"""
        try:
            with self.db.get_connection() as conn:
                conn.execute("UPDATE users SET role = ? WHERE user_id = ?",
                           (new_role.value, user_id))
                conn.commit()
            
            # Update active sessions
            with self.session_lock:
                for session in self.active_sessions.values():
                    if session.user_id == user_id:
                        session.role = new_role
                        session.permissions = self.role_permissions.get(new_role, set())
            
            logger.info(f"Promoted user {user_id} to {new_role.value}")
            
        except Exception as e:
            logger.error(f"Error promoting user {user_id}: {e}")
    
    def _terminate_user_sessions(self, user_id: str):
        """Terminate all sessions for a user"""
        with self.session_lock:
            sessions_to_remove = [
                session_id for session_id, session in self.active_sessions.items()
                if session.user_id == user_id
            ]
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
        
        logger.info(f"Terminated {len(sessions_to_remove)} sessions for user {user_id}")
    
    def _check_auto_quarantine(self, user_id: str):
        """Check if user should be automatically quarantined"""
        try:
            # Get recent security events
            with self.db.get_connection() as conn:
                cutoff = time.time() - (24 * 3600)  # Last 24 hours
                
                rows = conn.execute("""
                    SELECT event_type, severity, COUNT(*) as count
                    FROM security_events 
                    WHERE user_id = ? AND timestamp > ?
                    GROUP BY event_type, severity
                """, (user_id, cutoff)).fetchall()
                
                # Check quarantine triggers
                high_severity_count = sum(row['count'] for row in rows if row['severity'] >= 7)
                total_events = sum(row['count'] for row in rows)
                
                if high_severity_count >= 3:
                    self.quarantine_user(
                        user_id, QuarantineReason.SECURITY_VIOLATION, 8, 24,
                        {"trigger": "multiple_high_severity_events", "count": high_severity_count}
                    )
                elif total_events >= 10:
                    self.quarantine_user(
                        user_id, QuarantineReason.SUSPICIOUS_ACTIVITY, 5, 6,
                        {"trigger": "excessive_security_events", "count": total_events}
                    )
                    
        except Exception as e:
            logger.error(f"Error checking auto-quarantine for {user_id}: {e}")
    
    def generate_access_report(self, user_id: str = None) -> Dict[str, Any]:
        """Generate access control system report"""
        try:
            current_time = time.time()
            
            with self.db.get_connection() as conn:
                # System-wide statistics
                stats = {
                    'total_users': conn.execute("SELECT COUNT(*) FROM users").fetchone()[0],
                    'active_sessions': len(self.active_sessions),
                    'quarantined_users': conn.execute(
                        "SELECT COUNT(DISTINCT user_id) FROM quarantine_records WHERE active = TRUE"
                    ).fetchone()[0]
                }
                
                # Role distribution
                role_stats = {}
                for row in conn.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role"):
                    role_stats[row['role']] = row['count']
                
                # Recent security events
                recent_events = conn.execute("""
                    SELECT event_type, severity, COUNT(*) as count
                    FROM security_events 
                    WHERE timestamp > ?
                    GROUP BY event_type, severity
                    ORDER BY count DESC LIMIT 10
                """, (current_time - 7 * 24 * 3600,)).fetchall()
                
                report = {
                    'generation_time': current_time,
                    'system_statistics': stats,
                    'role_distribution': role_stats,
                    'recent_security_events': [dict(row) for row in recent_events],
                    'rate_limit_rules': {name: asdict(rule) for name, rule in self.rate_limit_rules.items()}
                }
                
                # User-specific information if requested
                if user_id:
                    user_data = self.db.get_user(user_id)
                    if user_data:
                        reputation_history = self.db.get_reputation_history(user_id, 30)
                        quarantine_status = self.check_user_quarantine(user_id)
                        
                        report['user_details'] = {
                            'user_id': user_id,
                            'role': user_data['role'],
                            'reputation_score': user_data['reputation_score'],
                            'security_score': user_data['security_score'],
                            'account_age_days': (current_time - user_data['created_at']) / (24 * 3600),
                            'last_activity_hours': (current_time - user_data['last_activity']) / 3600,
                            'reputation_changes_30d': len(reputation_history),
                            'quarantined': quarantine_status is not None
                        }
                
                return report
                
        except Exception as e:
            logger.error(f"Error generating access report: {e}")
            return {'error': str(e)}
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        session_timeout = 24 * 3600  # 24 hours
        
        with self.session_lock:
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if current_time - session.last_activity > session_timeout
            ]
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


def create_steam_deck_access_control() -> AccessControlSystem:
    """Create access control system optimized for Steam Deck shader sharing"""
    
    # Create with Steam Deck specific configuration
    system = AccessControlSystem()
    
    # Adjust rate limits for Steam Deck users
    system.rate_limit_rules.update({
        "shader_upload": RateLimitRule("shader_upload", 20, 3600, burst_allowance=5),
        "shader_download": RateLimitRule("shader_download", 200, 3600, burst_allowance=50),
        "optimization_data": RateLimitRule("optimization_data", 50, 3600),
    })
    
    return system


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create access control system
    ac_system = create_steam_deck_access_control()
    
    # Create test user
    user_id = "test_user_123"
    ac_system.create_user(user_id, "test@example.com", "password123", UserRole.VERIFIED)
    
    # Authenticate user
    token = ac_system.authenticate_user(user_id, "password123")
    if token:
        print(f"Authentication successful, token: {token[:32]}...")
        
        # Create session
        session = ac_system.create_session(token, "192.168.1.100", "SteamDeck/1.0")
        if session:
            print(f"Session created: {session.session_id}")
            
            # Test permissions
            can_upload = ac_system.check_permission(session.session_id, Permission.UPLOAD_SHADERS)
            can_admin = ac_system.check_permission(session.session_id, Permission.ADMIN_PANEL)
            
            print(f"Can upload shaders: {can_upload}")
            print(f"Can access admin: {can_admin}")
            
            # Test rate limiting
            for i in range(3):
                allowed = ac_system.check_rate_limit(session.session_id, "shader_upload")
                print(f"Upload attempt {i+1}: {'allowed' if allowed else 'rate limited'}")
            
            # Update reputation
            ac_system.update_reputation(user_id, ReputationAction.SHADER_UPLOAD, "Successful upload")
            ac_system.update_reputation(user_id, ReputationAction.SUCCESSFUL_VALIDATION, "Shader validated")
            
            # Generate report
            report = ac_system.generate_access_report(user_id)
            print(f"\nUser reputation: {report['user_details']['reputation_score']}")
            print(f"Account age: {report['user_details']['account_age_days']:.1f} days")
    
    else:
        print("Authentication failed")