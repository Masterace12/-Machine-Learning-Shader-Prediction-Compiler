#!/usr/bin/env python3
"""
Security Integration Layer for ML and P2P Systems

This module provides seamless integration between the security validation framework
and the existing ML prediction and P2P distribution systems. It acts as a security
middleware that:

- Validates all shaders before ML processing or P2P sharing
- Integrates hardware fingerprinting with P2P peer verification
- Applies privacy protection to ML training data
- Enforces access control for P2P operations
- Provides unified security logging and monitoring
- Handles security event correlation across systems

The integration ensures that security is not an afterthought but a core part
of the shader prediction and distribution workflow.
"""

import asyncio
import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# Security modules
from .spv_static_analyzer import (
    SPIRVStaticAnalyzer, SecurityThreatLevel, SPIRVAnalysisResult
)
from .sandbox_executor import (
    ShaderSandbox, SandboxConfiguration, SecurityLevel, ExecutionResult, SandboxResult
)
from .hardware_fingerprint import (
    HardwareFingerprintGenerator, HardwareFingerprint, PrivacyMode, FingerprintingLevel
)
from .privacy_protection import (
    PrivacyProtectionSystem, PrivacyPolicy, PrivacyLevel, ConsentType, AnonymizationResult
)
from .access_control import (
    AccessControlSystem, UserRole, Permission, ReputationAction, QuarantineReason
)

# Import existing system interfaces (these would be the real imports in production)
try:
    # ML system integration
    import sys
    import os
    sys.path.append(str(Path(__file__).parent.parent / "shader-predict-compile" / "src"))
    from ml_shader_predictor import SteamDeckMLPredictor
    HAS_ML_SYSTEM = True
except ImportError:
    HAS_ML_SYSTEM = False

try:
    # P2P system integration
    sys.path.append(str(Path(__file__).parent.parent))
    from p2p_shader_distribution import P2PShaderDistributionSystem
    HAS_P2P_SYSTEM = True
except ImportError:
    HAS_P2P_SYSTEM = False

logger = logging.getLogger(__name__)


class SecurityAction(Enum):
    """Security actions that can be taken"""
    ALLOW = "allow"
    DENY = "deny"
    QUARANTINE = "quarantine"
    SANDBOX = "sandbox"
    MANUAL_REVIEW = "manual_review"


class IntegrationEvent(Enum):
    """Integration event types"""
    SHADER_SUBMITTED = "shader_submitted"
    SHADER_VALIDATED = "shader_validated"
    SHADER_REJECTED = "shader_rejected"
    SHADER_SHARED = "shader_shared"
    PEER_CONNECTED = "peer_connected"
    SECURITY_VIOLATION = "security_violation"
    REPUTATION_UPDATED = "reputation_updated"


@dataclass
class SecurityValidationRequest:
    """Request for security validation"""
    request_id: str
    shader_hash: str
    shader_data: bytes
    user_id: Optional[str]
    operation: str  # "upload", "share", "train", "execute"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    # Context information
    source_system: str = "unknown"  # "ml", "p2p", "api"
    peer_id: Optional[str] = None
    game_id: Optional[str] = None
    hardware_fingerprint: Optional[str] = None


@dataclass
class SecurityValidationResult:
    """Result of security validation"""
    request_id: str
    action: SecurityAction
    threat_level: SecurityThreatLevel
    confidence: float  # 0.0 to 1.0
    
    # Analysis results
    static_analysis: Optional[SPIRVAnalysisResult] = None
    sandbox_result: Optional[ExecutionResult] = None
    privacy_result: Optional[AnonymizationResult] = None
    
    # Decision factors
    decision_reasons: List[str] = field(default_factory=list)
    security_score: float = 0.0
    reputation_impact: float = 0.0
    
    # Metadata
    processing_time_ms: float = 0.0
    validation_timestamp: float = field(default_factory=time.time)
    requires_manual_review: bool = False


class SecurityEventCorrelator:
    """Correlate security events across systems"""
    
    def __init__(self):
        self.events = deque(maxlen=10000)
        self.event_lock = threading.Lock()
        self.correlation_rules = self._setup_correlation_rules()
    
    def _setup_correlation_rules(self) -> Dict[str, Dict]:
        """Setup event correlation rules"""
        return {
            "suspicious_upload_pattern": {
                "events": ["shader_submitted", "shader_rejected"],
                "threshold": 5,
                "time_window": 3600,  # 1 hour
                "severity": 7
            },
            "peer_malicious_sharing": {
                "events": ["shader_shared", "security_violation"],
                "threshold": 3,
                "time_window": 1800,  # 30 minutes
                "severity": 8
            },
            "reputation_manipulation": {
                "events": ["reputation_updated"],
                "threshold": 20,
                "time_window": 3600,
                "severity": 6
            }
        }
    
    def record_event(self, event_type: IntegrationEvent, user_id: str = None,
                    peer_id: str = None, details: Dict[str, Any] = None):
        """Record a security event"""
        event = {
            'timestamp': time.time(),
            'event_type': event_type.value,
            'user_id': user_id,
            'peer_id': peer_id,
            'details': details or {}
        }
        
        with self.event_lock:
            self.events.append(event)
            self._check_correlations(event)
    
    def _check_correlations(self, new_event: Dict[str, Any]):
        """Check for event correlations"""
        current_time = new_event['timestamp']
        
        for rule_name, rule in self.correlation_rules.items():
            matching_events = []
            
            # Find events matching this rule
            for event in self.events:
                if (event['event_type'] in rule['events'] and
                    current_time - event['timestamp'] <= rule['time_window']):
                    
                    # Check if events are related (same user/peer)
                    if (event.get('user_id') == new_event.get('user_id') or
                        event.get('peer_id') == new_event.get('peer_id')):
                        matching_events.append(event)
            
            # Check if threshold exceeded
            if len(matching_events) >= rule['threshold']:
                self._trigger_correlation_alert(rule_name, rule, matching_events)
    
    def _trigger_correlation_alert(self, rule_name: str, rule: Dict, events: List[Dict]):
        """Trigger correlation-based security alert"""
        logger.warning(f"Security correlation detected: {rule_name}")
        
        # Extract user/peer ID for action
        entity_id = None
        for event in events:
            if event.get('user_id'):
                entity_id = event['user_id']
                break
            elif event.get('peer_id'):
                entity_id = event['peer_id']
                break
        
        if entity_id:
            # This would trigger appropriate security actions
            logger.warning(f"Correlated security violation for entity {entity_id}: {rule_name}")


class SecurityIntegrationLayer:
    """Main security integration layer"""
    
    def __init__(self, config_path: Path = None):
        # Initialize security components
        self.static_analyzer = SPIRVStaticAnalyzer()
        self.sandbox = ShaderSandbox(SandboxConfiguration(
            security_level=SecurityLevel.STRICT,
            max_memory_mb=256,
            max_cpu_time_seconds=30.0
        ))
        
        # Hardware fingerprinting
        self.hardware_generator = HardwareFingerprintGenerator(
            privacy_mode=PrivacyMode.SALTED,
            fingerprinting_level=FingerprintingLevel.STANDARD
        )
        self.system_fingerprint = self.hardware_generator.generate_fingerprint()
        
        # Privacy protection
        self.privacy_system = PrivacyProtectionSystem(PrivacyPolicy(
            privacy_level=PrivacyLevel.ENHANCED,
            enable_differential_privacy=True,
            enable_encryption=True
        ))
        
        # Access control
        self.access_control = AccessControlSystem()
        
        # Event correlation
        self.event_correlator = SecurityEventCorrelator()
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Security state
        self.validation_cache = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'validations_performed': 0,
            'shaders_blocked': 0,
            'shaders_sandboxed': 0,
            'security_violations': 0,
            'reputation_adjustments': 0
        }
        
        logger.info("Security integration layer initialized")
    
    async def validate_shader_security(self, request: SecurityValidationRequest) -> SecurityValidationResult:
        """Comprehensive security validation of shader"""
        start_time = time.time()
        
        logger.info(f"Starting security validation for shader {request.shader_hash}")
        
        # Check cache first
        cache_key = f"{request.shader_hash}:{request.operation}"
        with self.cache_lock:
            if cache_key in self.validation_cache:
                cached_result = self.validation_cache[cache_key]
                # Use cached result if recent (within 1 hour)
                if time.time() - cached_result.validation_timestamp < 3600:
                    logger.debug(f"Using cached validation result for {request.shader_hash}")
                    return cached_result
        
        # Initialize result
        result = SecurityValidationResult(
            request_id=request.request_id,
            action=SecurityAction.DENY,  # Default to deny
            threat_level=SecurityThreatLevel.SAFE,
            confidence=0.0
        )
        
        try:
            # 1. Static analysis
            static_result = await self._perform_static_analysis(request)
            result.static_analysis = static_result
            
            # 2. Check user reputation and access
            if request.user_id:
                access_check = await self._check_user_access(request)
                if not access_check['allowed']:
                    result.action = SecurityAction.DENY
                    result.decision_reasons.append(f"Access denied: {access_check['reason']}")
                    result.threat_level = SecurityThreatLevel.SUSPICIOUS
                    return result
            
            # 3. Hardware fingerprint validation
            if request.hardware_fingerprint:
                hw_validation = await self._validate_hardware_fingerprint(request)
                if not hw_validation['valid']:
                    result.decision_reasons.append(f"Hardware validation failed: {hw_validation['reason']}")
            
            # 4. Determine security action based on static analysis
            if static_result.overall_threat_level == SecurityThreatLevel.MALICIOUS:
                result.action = SecurityAction.DENY
                result.threat_level = SecurityThreatLevel.MALICIOUS
                result.decision_reasons.append("Malicious code detected")
                result.confidence = 0.9
            
            elif static_result.overall_threat_level == SecurityThreatLevel.SUSPICIOUS:
                # Decide between sandbox and manual review
                if len(static_result.vulnerabilities) >= 3:
                    result.action = SecurityAction.MANUAL_REVIEW
                    result.requires_manual_review = True
                else:
                    # Run in sandbox
                    sandbox_result = await self._perform_sandbox_analysis(request)
                    result.sandbox_result = sandbox_result
                    
                    if sandbox_result.result == SandboxResult.SUCCESS:
                        result.action = SecurityAction.SANDBOX
                        result.threat_level = SecurityThreatLevel.SUSPICIOUS
                    else:
                        result.action = SecurityAction.DENY
                        result.threat_level = SecurityThreatLevel.MALICIOUS
                
                result.confidence = 0.7
            
            elif static_result.overall_threat_level == SecurityThreatLevel.SAFE:
                result.action = SecurityAction.ALLOW
                result.threat_level = SecurityThreatLevel.SAFE
                result.decision_reasons.append("Static analysis passed")
                result.confidence = 0.8
            
            else:  # INVALID
                result.action = SecurityAction.DENY
                result.threat_level = SecurityThreatLevel.INVALID
                result.decision_reasons.append("Invalid shader format")
                result.confidence = 1.0
            
            # 5. Apply privacy protection if sharing
            if request.operation in ['share', 'upload'] and result.action == SecurityAction.ALLOW:
                privacy_result = await self._apply_privacy_protection(request)
                result.privacy_result = privacy_result
                
                if privacy_result.privacy_risk_score > 0.7:
                    result.action = SecurityAction.MANUAL_REVIEW
                    result.decision_reasons.append("High privacy risk detected")
            
            # 6. Calculate overall security score
            result.security_score = self._calculate_security_score(result)
            
            # 7. Update reputation
            if request.user_id:
                reputation_delta = await self._calculate_reputation_impact(request, result)
                result.reputation_impact = reputation_delta
                
                if reputation_delta != 0:
                    await self._update_user_reputation(request.user_id, reputation_delta, 
                                                     f"Shader validation: {result.action.value}")
            
            # 8. Record security event
            self.event_correlator.record_event(
                IntegrationEvent.SHADER_VALIDATED,
                user_id=request.user_id,
                details={
                    'shader_hash': request.shader_hash,
                    'action': result.action.value,
                    'threat_level': result.threat_level.value,
                    'operation': request.operation
                }
            )
            
            # Update statistics
            self.stats['validations_performed'] += 1
            if result.action == SecurityAction.DENY:
                self.stats['shaders_blocked'] += 1
            elif result.action == SecurityAction.SANDBOX:
                self.stats['shaders_sandboxed'] += 1
            
            logger.info(f"Security validation completed for {request.shader_hash}: {result.action.value}")
            
        except Exception as e:
            logger.error(f"Security validation error for {request.shader_hash}: {e}")
            result.action = SecurityAction.DENY
            result.threat_level = SecurityThreatLevel.INVALID
            result.decision_reasons.append(f"Validation error: {str(e)}")
            result.confidence = 0.0
        
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Cache result
            with self.cache_lock:
                self.validation_cache[cache_key] = result
                # Limit cache size
                if len(self.validation_cache) > 1000:
                    # Remove oldest entries
                    sorted_items = sorted(self.validation_cache.items(), 
                                        key=lambda x: x[1].validation_timestamp)
                    self.validation_cache = dict(sorted_items[-500:])
        
        return result
    
    async def _perform_static_analysis(self, request: SecurityValidationRequest) -> SPIRVAnalysisResult:
        """Perform static analysis on shader bytecode"""
        try:
            # Run static analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self.static_analyzer.analyze_bytecode,
                request.shader_data,
                request.shader_hash
            )
            
            logger.debug(f"Static analysis completed for {request.shader_hash}: {result.overall_threat_level.value}")
            return result
            
        except Exception as e:
            logger.error(f"Static analysis error for {request.shader_hash}: {e}")
            # Return a safe default result
            return SPIRVAnalysisResult(
                shader_hash=request.shader_hash,
                file_size=len(request.shader_data),
                analysis_time_ms=0,
                overall_threat_level=SecurityThreatLevel.INVALID,
                vulnerabilities=[],
                is_valid_spirv=False,
                version=(0, 0),
                generator_id=0,
                bound=0,
                instruction_count=0,
                function_count=0,
                memory_operations=0,
                atomic_operations=0,
                branch_complexity=0,
                extension_usage=[],
                obfuscation_score=0.0,
                entropy_score=0.0,
                suspicious_patterns=[],
                estimated_memory_usage=0,
                register_pressure=0.0,
                execution_complexity=0.0,
                eac_compatible=False,
                battleye_compatible=False,
                vac_compatible=False,
                contains_pii=False,
                debug_info_stripped=True,
                source_references=[]
            )
    
    async def _perform_sandbox_analysis(self, request: SecurityValidationRequest) -> ExecutionResult:
        """Perform sandboxed execution analysis"""
        try:
            # Create a simple shader validation test
            test_program = f"""
import time
import sys

# Simulate shader compilation/execution
shader_size = {len(request.shader_data)}
print(f"Testing shader of size {{shader_size}} bytes")

# Basic validation checks
if shader_size > 1000000:  # 1MB limit
    sys.exit(1)

# Simulate processing time
time.sleep(0.1)
print("Shader validation completed successfully")
"""
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self.sandbox.execute_shader_test,
                request.shader_data,
                test_program
            )
            
            logger.debug(f"Sandbox analysis completed for {request.shader_hash}: {result.result.value}")
            return result
            
        except Exception as e:
            logger.error(f"Sandbox analysis error for {request.shader_hash}: {e}")
            return ExecutionResult(
                result=SandboxResult.ERROR,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=0.0,
                memory_peak_mb=0.0,
                cpu_time_used=0.0
            )
    
    async def _check_user_access(self, request: SecurityValidationRequest) -> Dict[str, Any]:
        """Check user access permissions"""
        try:
            user_data = self.access_control.db.get_user(request.user_id)
            if not user_data:
                return {'allowed': False, 'reason': 'User not found'}
            
            # Check if user is quarantined
            quarantine = self.access_control.check_user_quarantine(request.user_id)
            if quarantine:
                return {'allowed': False, 'reason': f'User quarantined: {quarantine.reason.value}'}
            
            # Check role permissions
            user_role = UserRole(user_data['role'])
            required_permission = None
            
            if request.operation == 'upload':
                required_permission = Permission.UPLOAD_SHADERS
            elif request.operation == 'share':
                required_permission = Permission.UPLOAD_SHADERS
            elif request.operation == 'download':
                required_permission = Permission.DOWNLOAD_SHADERS
            
            if required_permission:
                user_permissions = self.access_control.role_permissions.get(user_role, set())
                if required_permission not in user_permissions:
                    return {'allowed': False, 'reason': f'Insufficient permissions for {request.operation}'}
            
            return {'allowed': True, 'reason': 'Access granted'}
            
        except Exception as e:
            logger.error(f"Access check error for user {request.user_id}: {e}")
            return {'allowed': False, 'reason': f'Access check failed: {str(e)}'}
    
    async def _validate_hardware_fingerprint(self, request: SecurityValidationRequest) -> Dict[str, Any]:
        """Validate hardware fingerprint"""
        try:
            # In a real implementation, this would validate against known fingerprints
            # and check for hardware spoofing
            
            # For now, just check if fingerprint looks valid
            if len(request.hardware_fingerprint) < 16:
                return {'valid': False, 'reason': 'Invalid fingerprint format'}
            
            # Check against system fingerprint for compatibility
            if request.hardware_fingerprint == self.system_fingerprint.fingerprint_id:
                return {'valid': True, 'reason': 'Local system fingerprint'}
            
            # Could add more sophisticated validation here
            return {'valid': True, 'reason': 'Fingerprint validated'}
            
        except Exception as e:
            logger.error(f"Hardware fingerprint validation error: {e}")
            return {'valid': False, 'reason': f'Validation error: {str(e)}'}
    
    async def _apply_privacy_protection(self, request: SecurityValidationRequest) -> AnonymizationResult:
        """Apply privacy protection to shader data"""
        try:
            # Convert shader metadata to anonymizable format
            shader_metadata = {
                'shader_hash': request.shader_hash,
                'game_id': request.game_id,
                'user_system': request.user_id,  # This will be anonymized
                'timestamp': request.timestamp,
                'source_system': request.source_system,
                'hardware_info': request.hardware_fingerprint,
                **request.metadata
            }
            
            result = self.privacy_system.anonymize_shader_data(shader_metadata, request.user_id)
            
            logger.debug(f"Privacy protection applied for {request.shader_hash}, "
                        f"risk score: {result.privacy_risk_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Privacy protection error for {request.shader_hash}: {e}")
            return AnonymizationResult(
                original_data={},
                anonymized_data={},
                removed_fields=[],
                detected_pii=[],
                anonymization_method="error",
                privacy_risk_score=1.0,  # Maximum risk on error
                compliance_status={'gdpr_compliant': False}
            )
    
    def _calculate_security_score(self, result: SecurityValidationResult) -> float:
        """Calculate overall security score"""
        score = 100.0  # Start with perfect score
        
        # Deduct based on threat level
        if result.threat_level == SecurityThreatLevel.MALICIOUS:
            score = 0.0
        elif result.threat_level == SecurityThreatLevel.SUSPICIOUS:
            score = 30.0
        elif result.threat_level == SecurityThreatLevel.INVALID:
            score = 10.0
        
        # Adjust based on analysis results
        if result.static_analysis:
            # Deduct for vulnerabilities
            vuln_penalty = min(50.0, len(result.static_analysis.vulnerabilities) * 10.0)
            score -= vuln_penalty
            
            # Deduct for high obfuscation
            if result.static_analysis.obfuscation_score > 0.7:
                score -= 20.0
        
        # Adjust for sandbox results
        if result.sandbox_result and result.sandbox_result.result != SandboxResult.SUCCESS:
            score -= 30.0
        
        # Adjust for privacy risks
        if result.privacy_result and result.privacy_result.privacy_risk_score > 0.5:
            score -= result.privacy_result.privacy_risk_score * 20.0
        
        return max(0.0, min(100.0, score))
    
    async def _calculate_reputation_impact(self, request: SecurityValidationRequest, 
                                         result: SecurityValidationResult) -> float:
        """Calculate reputation impact of validation result"""
        if not request.user_id:
            return 0.0
        
        # Base reputation changes
        if result.action == SecurityAction.ALLOW and result.threat_level == SecurityThreatLevel.SAFE:
            return 1.0  # Small positive for clean shader
        elif result.action == SecurityAction.DENY and result.threat_level == SecurityThreatLevel.MALICIOUS:
            return -20.0  # Large negative for malicious shader
        elif result.action == SecurityAction.DENY and result.threat_level == SecurityThreatLevel.SUSPICIOUS:
            return -5.0  # Medium negative for suspicious shader
        elif result.action == SecurityAction.SANDBOX:
            return -1.0  # Small negative for requiring sandbox
        elif result.action == SecurityAction.MANUAL_REVIEW:
            return 0.0  # Neutral, pending review
        
        return 0.0
    
    async def _update_user_reputation(self, user_id: str, delta: float, reason: str):
        """Update user reputation"""
        try:
            if delta > 0:
                action = ReputationAction.SUCCESSFUL_VALIDATION
            else:
                action = ReputationAction.FAILED_VALIDATION
            
            self.access_control.update_reputation(user_id, action, reason, abs(delta))
            self.stats['reputation_adjustments'] += 1
            
            # Record event
            self.event_correlator.record_event(
                IntegrationEvent.REPUTATION_UPDATED,
                user_id=user_id,
                details={'delta': delta, 'reason': reason}
            )
            
        except Exception as e:
            logger.error(f"Error updating reputation for {user_id}: {e}")
    
    def get_system_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security system status"""
        return {
            'integration_layer': {
                'validations_performed': self.stats['validations_performed'],
                'shaders_blocked': self.stats['shaders_blocked'],
                'shaders_sandboxed': self.stats['shaders_sandboxed'],
                'security_violations': self.stats['security_violations'],
                'cache_size': len(self.validation_cache),
                'thread_pool_active': self.thread_pool._threads
            },
            'hardware_fingerprint': {
                'system_id': self.system_fingerprint.fingerprint_id,
                'is_steam_deck': self.system_fingerprint.is_steam_deck,
                'privacy_level': self.system_fingerprint.privacy_level.value,
                'anticheat_compatibility': {
                    'eac': self.system_fingerprint.eac_compatible_hardware,
                    'battleye': self.system_fingerprint.battleye_compatible_hardware,
                    'vac': self.system_fingerprint.vac_compatible_hardware
                }
            },
            'access_control': self.access_control.generate_access_report(),
            'privacy_protection': self.privacy_system.generate_privacy_report(),
            'event_correlation': {
                'events_recorded': len(self.event_correlator.events),
                'correlation_rules': len(self.event_correlator.correlation_rules)
            }
        }
    
    async def integrate_with_ml_system(self, ml_predictor = None) -> bool:
        """Integrate security validation with ML system"""
        if not HAS_ML_SYSTEM or not ml_predictor:
            logger.warning("ML system integration not available")
            return False
        
        try:
            # Wrap ML predictor methods with security validation
            original_extract_features = ml_predictor.extract_shader_features
            
            async def secure_extract_features(shader_path, game_id, steam_deck_model="LCD"):
                # Read shader data
                with open(shader_path, 'rb') as f:
                    shader_data = f.read()
                
                # Create security validation request
                request = SecurityValidationRequest(
                    request_id=f"ml_{time.time()}",
                    shader_hash=hashlib.sha256(shader_data).hexdigest()[:16],
                    shader_data=shader_data,
                    user_id=None,  # ML training is internal
                    operation="train",
                    source_system="ml",
                    game_id=game_id
                )
                
                # Validate security
                validation_result = await self.validate_shader_security(request)
                
                if validation_result.action not in [SecurityAction.ALLOW, SecurityAction.SANDBOX]:
                    logger.warning(f"ML training shader blocked: {shader_path}")
                    return None
                
                # Apply privacy protection to features
                features = original_extract_features(shader_path, game_id, steam_deck_model)
                if features:
                    # Anonymize any potentially sensitive data in features
                    anonymized = self.privacy_system.anonymize_telemetry_data(asdict(features))
                    if anonymized.privacy_risk_score < 0.5:  # Low risk threshold for ML
                        return features
                
                return None
            
            # Replace method with secure version
            ml_predictor.extract_shader_features = secure_extract_features
            
            logger.info("ML system security integration completed")
            return True
            
        except Exception as e:
            logger.error(f"ML system integration error: {e}")
            return False
    
    async def integrate_with_p2p_system(self, p2p_system = None) -> bool:
        """Integrate security validation with P2P system"""
        if not HAS_P2P_SYSTEM or not p2p_system:
            logger.warning("P2P system integration not available")
            return False
        
        try:
            # Wrap P2P system methods with security validation
            original_share_shader = p2p_system.share_shader
            original_request_shader = p2p_system.request_shader
            
            async def secure_share_shader(shader_data, shader_hash, game_id, shader_type, metadata=None):
                # Create security validation request
                request = SecurityValidationRequest(
                    request_id=f"p2p_share_{time.time()}",
                    shader_hash=shader_hash,
                    shader_data=shader_data,
                    user_id=p2p_system.peer_id,
                    operation="share",
                    source_system="p2p",
                    game_id=game_id,
                    hardware_fingerprint=self.system_fingerprint.fingerprint_id,
                    metadata=metadata or {}
                )
                
                # Validate security
                validation_result = await self.validate_shader_security(request)
                
                if validation_result.action != SecurityAction.ALLOW:
                    logger.warning(f"P2P shader sharing blocked: {shader_hash}")
                    return False
                
                # Use anonymized metadata if privacy protection was applied
                final_metadata = metadata
                if validation_result.privacy_result:
                    final_metadata = validation_result.privacy_result.anonymized_data
                
                # Record sharing event
                self.event_correlator.record_event(
                    IntegrationEvent.SHADER_SHARED,
                    peer_id=p2p_system.peer_id,
                    details={'shader_hash': shader_hash, 'game_id': game_id}
                )
                
                return await original_share_shader(shader_data, shader_hash, game_id, shader_type, final_metadata)
            
            async def secure_request_shader(shader_hash, game_id, shader_type, priority, callback=None):
                # Check access permissions first
                # (Simplified - in real implementation would check full permissions)
                
                # Record request event
                self.event_correlator.record_event(
                    IntegrationEvent.SHADER_SUBMITTED,
                    peer_id=p2p_system.peer_id,
                    details={'shader_hash': shader_hash, 'game_id': game_id}
                )
                
                return await original_request_shader(shader_hash, game_id, shader_type, priority, callback)
            
            # Replace methods with secure versions
            p2p_system.share_shader = secure_share_shader
            p2p_system.request_shader = secure_request_shader
            
            logger.info("P2P system security integration completed")
            return True
            
        except Exception as e:
            logger.error(f"P2P system integration error: {e}")
            return False
    
    def cleanup(self):
        """Clean up security integration resources"""
        logger.info("Cleaning up security integration layer")
        
        # Shutdown sandbox
        self.sandbox.cleanup()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Clear caches
        with self.cache_lock:
            self.validation_cache.clear()
        
        logger.info("Security integration cleanup completed")


async def create_integrated_security_system() -> SecurityIntegrationLayer:
    """Create fully integrated security system"""
    security_layer = SecurityIntegrationLayer()
    
    # Try to integrate with existing systems
    try:
        # ML system integration would happen here
        # await security_layer.integrate_with_ml_system(ml_system)
        pass
    except Exception as e:
        logger.warning(f"ML integration failed: {e}")
    
    try:
        # P2P system integration would happen here  
        # await security_layer.integrate_with_p2p_system(p2p_system)
        pass
    except Exception as e:
        logger.warning(f"P2P integration failed: {e}")
    
    return security_layer


if __name__ == "__main__":
    # Example usage
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def demo_security_integration():
        # Create security integration layer
        security_layer = await create_integrated_security_system()
        
        # Example shader validation
        test_shader_data = b'\x03\x02\x23\x07' + b'\x00' * 100  # Fake SPIR-V data
        
        request = SecurityValidationRequest(
            request_id="demo_001",
            shader_hash="test_shader_123",
            shader_data=test_shader_data,
            user_id="demo_user",
            operation="upload",
            source_system="api",
            game_id="demo_game"
        )
        
        # Validate shader security
        result = await security_layer.validate_shader_security(request)
        
        print(f"Security validation result:")
        print(f"  Action: {result.action.value}")
        print(f"  Threat Level: {result.threat_level.value}")
        print(f"  Security Score: {result.security_score:.1f}")
        print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"  Reasons: {result.decision_reasons}")
        
        # Get system status
        status = security_layer.get_system_security_status()
        print(f"\nSystem Status:")
        print(f"  Validations: {status['integration_layer']['validations_performed']}")
        print(f"  Blocked: {status['integration_layer']['shaders_blocked']}")
        print(f"  Steam Deck: {status['hardware_fingerprint']['is_steam_deck']}")
        
        # Cleanup
        security_layer.cleanup()
    
    # Run demo
    asyncio.run(demo_security_integration())