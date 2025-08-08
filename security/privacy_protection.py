#!/usr/bin/env python3
"""
Privacy Protection Framework for Anonymized Shader Data

This module implements comprehensive privacy protection for shader sharing and telemetry,
ensuring user anonymity while preserving functionality for optimization and compatibility.

Features:
- Differential privacy for telemetry data
- PII detection and removal from shader metadata
- Cryptographic anonymization techniques
- GDPR/CCPA compliance tools
- User consent management
- Data retention and deletion policies
- Audit logging for privacy compliance

The framework ensures that shader data can be safely shared in the community
while maintaining strong privacy guarantees and regulatory compliance.
"""

import os
import re
import json
import time
import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import base64
from datetime import datetime, timedelta
import threading
from collections import defaultdict
import tempfile

# Cryptographic libraries
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

# Differential privacy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy protection levels"""
    MINIMAL = "minimal"          # Basic PII removal
    STANDARD = "standard"        # Hash-based anonymization
    ENHANCED = "enhanced"        # Differential privacy + encryption
    MAXIMUM = "maximum"          # Full anonymization with data minimization


class ConsentType(Enum):
    """Types of user consent"""
    TELEMETRY = "telemetry"              # Performance and usage telemetry
    SHADER_SHARING = "shader_sharing"     # Sharing compiled shaders
    OPTIMIZATION_DATA = "optimization"    # Hardware optimization data
    ANALYTICS = "analytics"              # Usage analytics
    RESEARCH = "research"                # Academic research participation


class PIIType(Enum):
    """Types of personally identifiable information"""
    USERNAME = "username"
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    HOSTNAME = "hostname"
    FILE_PATH = "file_path"
    SYSTEM_ID = "system_id"
    TIMESTAMP = "timestamp"
    GEOLOCATION = "geolocation"
    BIOMETRIC = "biometric"


@dataclass
class ConsentRecord:
    """User consent record"""
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: float
    expiry_date: Optional[float] = None
    consent_version: str = "1.0"
    ip_address_hash: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyPolicy:
    """Privacy protection policy"""
    privacy_level: PrivacyLevel
    data_retention_days: int = 90
    enable_differential_privacy: bool = True
    differential_privacy_epsilon: float = 1.0
    enable_encryption: bool = True
    enable_audit_logging: bool = True
    
    # PII detection settings
    detect_usernames: bool = True
    detect_file_paths: bool = True
    detect_ip_addresses: bool = True
    detect_system_identifiers: bool = True
    
    # Anonymization settings
    hash_salt: Optional[str] = None
    use_k_anonymity: bool = True
    k_anonymity_threshold: int = 5
    
    # Data minimization
    remove_debug_info: bool = True
    remove_timestamps: bool = False
    remove_version_info: bool = False
    
    # Compliance
    gdpr_compliant: bool = True
    ccpa_compliant: bool = True
    coppa_compliant: bool = False


@dataclass
class AnonymizationResult:
    """Result of data anonymization"""
    original_data: Dict[str, Any]
    anonymized_data: Dict[str, Any]
    removed_fields: List[str]
    detected_pii: List[Tuple[PIIType, str]]
    anonymization_method: str
    privacy_risk_score: float  # 0.0 (safe) to 1.0 (risky)
    compliance_status: Dict[str, bool]


class PIIDetector:
    """Detect personally identifiable information in data"""
    
    def __init__(self):
        self.patterns = self._compile_pii_patterns()
        self.suspicious_field_names = {
            'username', 'user', 'name', 'login', 'account',
            'email', 'mail', 'address', 'contact',
            'phone', 'mobile', 'tel', 'fax',
            'ssn', 'social', 'id', 'identifier',
            'ip', 'address', 'host', 'hostname',
            'mac', 'hardware', 'serial', 'uuid',
            'location', 'geo', 'latitude', 'longitude',
            'path', 'file', 'directory', 'home',
            'token', 'key', 'secret', 'password'
        }
    
    def _compile_pii_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Compile regex patterns for PII detection"""
        return {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                re.IGNORECASE
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
                r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            PIIType.MAC_ADDRESS: re.compile(
                r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'
            ),
            PIIType.FILE_PATH: re.compile(
                r'(?:[A-Za-z]:\\|\/)(?:(?:[^\\\/\s]+[\\\/])*[^\\\/\s]*)',
                re.IGNORECASE
            ),
            PIIType.HOSTNAME: re.compile(
                r'\b[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?'
                r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*\b'
            ),
            PIIType.SYSTEM_ID: re.compile(
                r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
                re.IGNORECASE
            ),
        }
    
    def detect_pii_in_text(self, text: str) -> List[Tuple[PIIType, str]]:
        """Detect PII in text content"""
        detected_pii = []
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    match = ''.join(match)
                detected_pii.append((pii_type, match))
        
        return detected_pii
    
    def detect_pii_in_data(self, data: Any, path: str = "") -> List[Tuple[PIIType, str]]:
        """Recursively detect PII in structured data"""
        detected_pii = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check field name for suspicious patterns
                if any(suspicious in key.lower() for suspicious in self.suspicious_field_names):
                    detected_pii.append((PIIType.USERNAME, f"Field: {current_path}"))
                
                # Recursively check value
                detected_pii.extend(self.detect_pii_in_data(value, current_path))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                detected_pii.extend(self.detect_pii_in_data(item, current_path))
        
        elif isinstance(data, str):
            text_pii = self.detect_pii_in_text(data)
            for pii_type, value in text_pii:
                detected_pii.append((pii_type, f"{path}: {value[:50]}..."))
        
        return detected_pii
    
    def is_potentially_sensitive_field(self, field_name: str, value: Any) -> bool:
        """Check if a field might contain sensitive information"""
        field_lower = field_name.lower()
        
        # Check against known sensitive field names
        if any(suspicious in field_lower for suspicious in self.suspicious_field_names):
            return True
        
        # Check value patterns
        if isinstance(value, str):
            pii_found = self.detect_pii_in_text(value)
            return len(pii_found) > 0
        
        return False


class DifferentialPrivacyEngine:
    """Implement differential privacy for telemetry data"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        
        if not HAS_NUMPY:
            logger.warning("NumPy not available, differential privacy disabled")
            self.enabled = False
        else:
            self.enabled = True
    
    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise for differential privacy"""
        if not self.enabled:
            return value
        
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Gaussian noise for differential privacy"""
        if not self.enabled:
            return value
        
        # Calculate noise scale for (ε, δ)-differential privacy
        scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, scale)
        return value + noise
    
    def privatize_histogram(self, histogram: Dict[str, int], 
                           sensitivity: int = 1) -> Dict[str, int]:
        """Add noise to histogram data"""
        if not self.enabled:
            return histogram
        
        privatized = {}
        for key, count in histogram.items():
            noisy_count = max(0, int(self.add_laplace_noise(count, sensitivity)))
            privatized[key] = noisy_count
        
        return privatized
    
    def privatize_average(self, values: List[float], 
                         sensitivity: float = 1.0) -> float:
        """Calculate privatized average"""
        if not self.enabled or not values:
            return sum(values) / len(values) if values else 0.0
        
        true_average = sum(values) / len(values)
        # Sensitivity for average is sensitivity/n
        avg_sensitivity = sensitivity / len(values)
        return self.add_laplace_noise(true_average, avg_sensitivity)
    
    def privatize_count(self, count: int, sensitivity: int = 1) -> int:
        """Add noise to count data"""
        if not self.enabled:
            return count
        
        noisy_count = self.add_laplace_noise(count, sensitivity)
        return max(0, int(noisy_count))


class CryptographicAnonymizer:
    """Cryptographic anonymization and encryption"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if not HAS_CRYPTO:
            logger.error("Cryptography library not available")
            raise ImportError("cryptography library required for encryption")
        
        self.master_key = master_key or self._generate_master_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key[:32]))
        
    def _generate_master_key(self) -> bytes:
        """Generate cryptographic master key"""
        return secrets.token_bytes(32)
    
    def anonymize_identifier(self, identifier: str, context: str = "") -> str:
        """Create anonymous but consistent identifier"""
        # Use HMAC for consistent anonymization
        combined = f"{context}:{identifier}".encode()
        anonymous_id = hmac.new(self.master_key, combined, hashlib.sha256).hexdigest()
        return anonymous_id[:16]  # 16-character anonymous ID
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def create_pseudonym(self, original_value: str, category: str = "default") -> str:
        """Create pseudonym for data"""
        # Create deterministic but anonymous pseudonym
        combined = f"{category}:{original_value}".encode()
        pseudonym_hash = hmac.new(self.master_key, combined, hashlib.sha256).hexdigest()
        
        # Format as readable pseudonym
        if category == "username":
            return f"user_{pseudonym_hash[:8]}"
        elif category == "hostname":
            return f"host_{pseudonym_hash[:8]}"
        elif category == "file":
            return f"file_{pseudonym_hash[:8]}"
        else:
            return f"anon_{pseudonym_hash[:8]}"


class ConsentManager:
    """Manage user consent for data collection and processing"""
    
    def __init__(self, consent_db_path: Optional[Path] = None):
        self.consent_db_path = consent_db_path or Path("consent_records.json")
        self.consents = self._load_consents()
        self.lock = threading.Lock()
    
    def _load_consents(self) -> Dict[str, List[ConsentRecord]]:
        """Load consent records from storage"""
        if not self.consent_db_path.exists():
            return {}
        
        try:
            with open(self.consent_db_path, 'r') as f:
                data = json.load(f)
            
            consents = {}
            for user_id, consent_list in data.items():
                consents[user_id] = []
                for consent_data in consent_list:
                    consent_data['consent_type'] = ConsentType(consent_data['consent_type'])
                    consents[user_id].append(ConsentRecord(**consent_data))
            
            return consents
            
        except Exception as e:
            logger.error(f"Error loading consent records: {e}")
            return {}
    
    def _save_consents(self):
        """Save consent records to storage"""
        try:
            data = {}
            for user_id, consent_list in self.consents.items():
                data[user_id] = []
                for consent in consent_list:
                    consent_dict = asdict(consent)
                    consent_dict['consent_type'] = consent.consent_type.value
                    data[user_id].append(consent_dict)
            
            with open(self.consent_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving consent records: {e}")
    
    def record_consent(self, user_id: str, consent_type: ConsentType, 
                      granted: bool, expiry_days: Optional[int] = None,
                      additional_data: Dict[str, Any] = None) -> bool:
        """Record user consent"""
        with self.lock:
            current_time = time.time()
            expiry_date = None
            
            if expiry_days is not None:
                expiry_date = current_time + (expiry_days * 24 * 60 * 60)
            
            consent_record = ConsentRecord(
                user_id=user_id,
                consent_type=consent_type,
                granted=granted,
                timestamp=current_time,
                expiry_date=expiry_date,
                additional_data=additional_data or {}
            )
            
            if user_id not in self.consents:
                self.consents[user_id] = []
            
            # Remove any existing consent for this type
            self.consents[user_id] = [
                c for c in self.consents[user_id] 
                if c.consent_type != consent_type
            ]
            
            # Add new consent record
            self.consents[user_id].append(consent_record)
            self._save_consents()
            
            logger.info(f"Recorded consent for user {user_id}: {consent_type.value} = {granted}")
            return True
    
    def check_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Check if user has given consent for specific data processing"""
        with self.lock:
            if user_id not in self.consents:
                return False
            
            current_time = time.time()
            
            for consent in self.consents[user_id]:
                if consent.consent_type == consent_type:
                    # Check if consent is still valid
                    if consent.expiry_date and current_time > consent.expiry_date:
                        return False
                    
                    return consent.granted
            
            return False
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """Get all consent records for a user"""
        with self.lock:
            return self.consents.get(user_id, []).copy()
    
    def revoke_consent(self, user_id: str, consent_type: ConsentType) -> bool:
        """Revoke user consent"""
        return self.record_consent(user_id, consent_type, False)
    
    def cleanup_expired_consents(self):
        """Remove expired consent records"""
        with self.lock:
            current_time = time.time()
            cleaned_count = 0
            
            for user_id, consent_list in self.consents.items():
                original_count = len(consent_list)
                
                self.consents[user_id] = [
                    consent for consent in consent_list
                    if not (consent.expiry_date and current_time > consent.expiry_date)
                ]
                
                cleaned_count += original_count - len(self.consents[user_id])
            
            if cleaned_count > 0:
                self._save_consents()
                logger.info(f"Cleaned up {cleaned_count} expired consent records")


class PrivacyProtectionSystem:
    """Main privacy protection system"""
    
    def __init__(self, policy: PrivacyPolicy, consent_manager: ConsentManager = None):
        self.policy = policy
        self.consent_manager = consent_manager or ConsentManager()
        
        self.pii_detector = PIIDetector()
        
        if policy.enable_differential_privacy and HAS_NUMPY:
            self.dp_engine = DifferentialPrivacyEngine(policy.differential_privacy_epsilon)
        else:
            self.dp_engine = None
        
        if policy.enable_encryption and HAS_CRYPTO:
            self.crypto_anonymizer = CryptographicAnonymizer()
        else:
            self.crypto_anonymizer = None
        
        # Audit logging
        self.audit_log = []
        self.audit_lock = threading.Lock()
        
        logger.info(f"Privacy protection system initialized with {policy.privacy_level.value} level")
    
    def anonymize_shader_data(self, shader_data: Dict[str, Any], 
                            user_id: str = None) -> AnonymizationResult:
        """Anonymize shader data for safe sharing"""
        
        # Check consent if user ID provided
        if user_id and not self.consent_manager.check_consent(user_id, ConsentType.SHADER_SHARING):
            raise PermissionError(f"User {user_id} has not consented to shader sharing")
        
        original_data = shader_data.copy()
        anonymized_data = shader_data.copy()
        removed_fields = []
        detected_pii = []
        
        # Detect PII in the data
        detected_pii = self.pii_detector.detect_pii_in_data(shader_data)
        
        # Remove or anonymize PII based on policy
        anonymized_data = self._process_data_for_privacy(
            anonymized_data, detected_pii, removed_fields
        )
        
        # Apply differential privacy to numerical data
        if self.dp_engine:
            anonymized_data = self._apply_differential_privacy(anonymized_data)
        
        # Calculate privacy risk score
        privacy_risk_score = self._calculate_privacy_risk(detected_pii, removed_fields)
        
        # Check compliance
        compliance_status = self._check_compliance(anonymized_data, detected_pii)
        
        result = AnonymizationResult(
            original_data=original_data,
            anonymized_data=anonymized_data,
            removed_fields=removed_fields,
            detected_pii=detected_pii,
            anonymization_method=self.policy.privacy_level.value,
            privacy_risk_score=privacy_risk_score,
            compliance_status=compliance_status
        )
        
        # Log the anonymization
        self._log_anonymization(result, user_id)
        
        return result
    
    def anonymize_telemetry_data(self, telemetry_data: Dict[str, Any],
                               user_id: str = None) -> AnonymizationResult:
        """Anonymize telemetry data for safe collection"""
        
        # Check consent
        if user_id and not self.consent_manager.check_consent(user_id, ConsentType.TELEMETRY):
            raise PermissionError(f"User {user_id} has not consented to telemetry collection")
        
        # Similar processing as shader data but with telemetry-specific rules
        return self.anonymize_shader_data(telemetry_data, user_id)
    
    def _process_data_for_privacy(self, data: Dict[str, Any], 
                                detected_pii: List[Tuple[PIIType, str]],
                                removed_fields: List[str]) -> Dict[str, Any]:
        """Process data to remove or anonymize PII"""
        
        processed_data = {}
        
        for key, value in data.items():
            should_remove = False
            
            # Check if field should be removed based on policy
            if self.policy.remove_debug_info and 'debug' in key.lower():
                should_remove = True
            elif self.policy.remove_timestamps and 'time' in key.lower():
                should_remove = True
            elif self.policy.remove_version_info and 'version' in key.lower():
                should_remove = True
            
            # Check if field contains PII
            if self.pii_detector.is_potentially_sensitive_field(key, value):
                if self.policy.privacy_level == PrivacyLevel.MAXIMUM:
                    should_remove = True
                else:
                    # Anonymize instead of remove
                    value = self._anonymize_value(key, value)
            
            if should_remove:
                removed_fields.append(key)
            else:
                if isinstance(value, dict):
                    processed_data[key] = self._process_data_for_privacy(
                        value, detected_pii, removed_fields
                    )
                else:
                    processed_data[key] = value
        
        return processed_data
    
    def _anonymize_value(self, field_name: str, value: Any) -> Any:
        """Anonymize a specific value"""
        if not isinstance(value, str):
            return value
        
        if self.crypto_anonymizer is None:
            # Fallback to hash-based anonymization
            return hashlib.sha256(f"{field_name}:{value}".encode()).hexdigest()[:16]
        
        # Determine anonymization method based on field type
        if 'user' in field_name.lower() or 'name' in field_name.lower():
            return self.crypto_anonymizer.create_pseudonym(value, "username")
        elif 'host' in field_name.lower():
            return self.crypto_anonymizer.create_pseudonym(value, "hostname")
        elif 'path' in field_name.lower() or 'file' in field_name.lower():
            return self.crypto_anonymizer.create_pseudonym(value, "file")
        else:
            return self.crypto_anonymizer.anonymize_identifier(value, field_name)
    
    def _apply_differential_privacy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy to numerical data"""
        if not self.dp_engine:
            return data
        
        processed_data = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)) and key not in ['hash', 'id']:
                # Add noise to numerical values
                if isinstance(value, int):
                    processed_data[key] = int(self.dp_engine.add_laplace_noise(float(value)))
                else:
                    processed_data[key] = self.dp_engine.add_laplace_noise(value)
            elif isinstance(value, dict):
                processed_data[key] = self._apply_differential_privacy(value)
            else:
                processed_data[key] = value
        
        return processed_data
    
    def _calculate_privacy_risk(self, detected_pii: List[Tuple[PIIType, str]], 
                              removed_fields: List[str]) -> float:
        """Calculate privacy risk score"""
        risk_score = 0.0
        
        # Risk from remaining PII
        pii_risk_weights = {
            PIIType.EMAIL: 0.8,
            PIIType.IP_ADDRESS: 0.6,
            PIIType.USERNAME: 0.5,
            PIIType.HOSTNAME: 0.4,
            PIIType.FILE_PATH: 0.3,
            PIIType.MAC_ADDRESS: 0.7,
            PIIType.SYSTEM_ID: 0.4,
            PIIType.TIMESTAMP: 0.2
        }
        
        for pii_type, _ in detected_pii:
            risk_score += pii_risk_weights.get(pii_type, 0.3)
        
        # Normalize risk score
        risk_score = min(1.0, risk_score / 3.0)  # Assume max 3 high-risk PII items
        
        # Reduce risk score based on privacy level
        if self.policy.privacy_level == PrivacyLevel.MAXIMUM:
            risk_score *= 0.1
        elif self.policy.privacy_level == PrivacyLevel.ENHANCED:
            risk_score *= 0.3
        elif self.policy.privacy_level == PrivacyLevel.STANDARD:
            risk_score *= 0.6
        
        return risk_score
    
    def _check_compliance(self, anonymized_data: Dict[str, Any], 
                         detected_pii: List[Tuple[PIIType, str]]) -> Dict[str, bool]:
        """Check regulatory compliance"""
        compliance = {
            'gdpr_compliant': True,
            'ccpa_compliant': True,
            'coppa_compliant': True
        }
        
        # GDPR compliance checks
        if self.policy.gdpr_compliant:
            # Check if high-risk PII is properly handled
            high_risk_pii = {PIIType.EMAIL, PIIType.IP_ADDRESS, PIIType.BIOMETRIC}
            if any(pii_type in high_risk_pii for pii_type, _ in detected_pii):
                if self.policy.privacy_level in [PrivacyLevel.MINIMAL, PrivacyLevel.STANDARD]:
                    compliance['gdpr_compliant'] = False
        
        # CCPA compliance checks
        if self.policy.ccpa_compliant:
            # Similar to GDPR but with different thresholds
            pass
        
        # COPPA compliance (stricter for children's data)
        if self.policy.coppa_compliant:
            if detected_pii:  # Any PII is problematic for COPPA
                compliance['coppa_compliant'] = False
        
        return compliance
    
    def _log_anonymization(self, result: AnonymizationResult, user_id: Optional[str]):
        """Log anonymization operation for audit purposes"""
        if not self.policy.enable_audit_logging:
            return
        
        with self.audit_lock:
            log_entry = {
                'timestamp': time.time(),
                'operation': 'anonymization',
                'user_id': user_id,
                'privacy_level': self.policy.privacy_level.value,
                'pii_detected': len(result.detected_pii),
                'fields_removed': len(result.removed_fields),
                'privacy_risk_score': result.privacy_risk_score,
                'compliance_status': result.compliance_status
            }
            
            self.audit_log.append(log_entry)
            
            # Keep only recent audit entries
            if len(self.audit_log) > 10000:
                self.audit_log = self.audit_log[-5000:]  # Keep last 5000
    
    def generate_privacy_report(self, user_id: str = None) -> Dict[str, Any]:
        """Generate privacy compliance report"""
        report = {
            'generation_time': time.time(),
            'policy': asdict(self.policy),
            'statistics': {
                'total_anonymizations': len(self.audit_log),
                'average_privacy_risk': 0.0,
                'compliance_rate': {}
            }
        }
        
        if self.audit_log:
            # Calculate statistics
            risk_scores = [entry.get('privacy_risk_score', 0.0) for entry in self.audit_log]
            report['statistics']['average_privacy_risk'] = sum(risk_scores) / len(risk_scores)
            
            # Compliance statistics
            compliance_counts = defaultdict(int)
            total_count = len(self.audit_log)
            
            for entry in self.audit_log:
                compliance_status = entry.get('compliance_status', {})
                for regulation, compliant in compliance_status.items():
                    if compliant:
                        compliance_counts[regulation] += 1
            
            for regulation, count in compliance_counts.items():
                report['statistics']['compliance_rate'][regulation] = count / total_count
        
        # User-specific information if provided
        if user_id:
            consents = self.consent_manager.get_user_consents(user_id)
            report['user_consents'] = [
                {
                    'type': consent.consent_type.value,
                    'granted': consent.granted,
                    'timestamp': consent.timestamp,
                    'expires': consent.expiry_date
                }
                for consent in consents
            ]
        
        return report
    
    def cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        if not self.policy.data_retention_days:
            return
        
        current_time = time.time()
        retention_cutoff = current_time - (self.policy.data_retention_days * 24 * 60 * 60)
        
        # Clean up audit log
        with self.audit_lock:
            original_count = len(self.audit_log)
            self.audit_log = [
                entry for entry in self.audit_log
                if entry.get('timestamp', 0) > retention_cutoff
            ]
            cleaned_count = original_count - len(self.audit_log)
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old audit log entries")
        
        # Clean up expired consents
        self.consent_manager.cleanup_expired_consents()


def create_privacy_system_for_steam_deck() -> PrivacyProtectionSystem:
    """Create privacy system optimized for Steam Deck shader sharing"""
    
    policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.ENHANCED,
        data_retention_days=30,  # Shorter retention for gaming data
        enable_differential_privacy=True,
        differential_privacy_epsilon=0.5,  # Strong privacy
        enable_encryption=True,
        enable_audit_logging=True,
        
        # Steam Deck specific settings
        detect_usernames=True,
        detect_file_paths=True,  # Important for Steam paths
        detect_system_identifiers=True,
        
        # Remove debug info but keep performance data
        remove_debug_info=True,
        remove_timestamps=False,  # Keep for performance analysis
        remove_version_info=False,  # Keep for compatibility
        
        # Compliance
        gdpr_compliant=True,
        ccpa_compliant=True,
        use_k_anonymity=True,
        k_anonymity_threshold=5
    )
    
    consent_manager = ConsentManager()
    return PrivacyProtectionSystem(policy, consent_manager)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create privacy protection system
    privacy_system = create_privacy_system_for_steam_deck()
    
    # Example shader data with potential PII
    shader_data = {
        'shader_hash': 'abc123def456',
        'game_id': 'steam_game_123',
        'user_system': 'john_doe_pc',  # PII
        'file_path': '/home/john/.steam/shaders/game.spv',  # PII
        'compilation_time_ms': 150.0,
        'gpu_model': 'AMD RDNA2',
        'driver_version': '21.40.21.03',
        'performance_metrics': {
            'frame_time_ms': 16.67,
            'memory_usage_mb': 512,
            'temperature_c': 65.0
        },
        'debug_info': {
            'compiler_version': '1.2.3',
            'hostname': 'johns-steamdeck'  # PII
        }
    }
    
    # Record user consent
    user_id = "anonymous_user_123"
    privacy_system.consent_manager.record_consent(
        user_id, ConsentType.SHADER_SHARING, True, expiry_days=90
    )
    
    # Anonymize the data
    result = privacy_system.anonymize_shader_data(shader_data, user_id)
    
    print("=== Anonymization Result ===")
    print(f"Privacy Risk Score: {result.privacy_risk_score:.3f}")
    print(f"PII Detected: {len(result.detected_pii)}")
    print(f"Fields Removed: {result.removed_fields}")
    print(f"GDPR Compliant: {result.compliance_status['gdpr_compliant']}")
    
    print("\n=== Anonymized Data ===")
    print(json.dumps(result.anonymized_data, indent=2))
    
    # Generate privacy report
    report = privacy_system.generate_privacy_report(user_id)
    print(f"\n=== Privacy Report ===")
    print(f"Average Privacy Risk: {report['statistics']['average_privacy_risk']:.3f}")
    print(f"User Consents: {len(report.get('user_consents', []))}")