"""
Comprehensive Security Validation Framework for Shader Prediction Compilation

This package provides robust security validation for community shader sharing,
ensuring safe deployment on Steam Deck while maintaining compatibility with
anti-cheat systems and protecting user privacy.

Main Components:
- SPIR-V static analysis for malware detection
- Sandboxed execution environment
- Hardware fingerprinting for compatibility
- Privacy protection with anonymization
- Access control with reputation system
- Anti-cheat compatibility checking
- Security integration layer
- Comprehensive testing suite

Example Usage:
    >>> from security import create_secure_shader_system
    >>> security_system = await create_secure_shader_system()
    >>> result = await security_system.validate_shader(shader_data, user_id)
"""

__version__ = "1.0.0"
__author__ = "Claude (Anthropic)"

# Core security components
from .spv_static_analyzer import (
    SPIRVStaticAnalyzer,
    SecurityThreatLevel,
    SPIRVAnalysisResult,
    create_test_analyzer
)

from .sandbox_executor import (
    ShaderSandbox,
    SandboxConfiguration,
    SecurityLevel,
    SandboxResult,
    ExecutionResult,
    ShaderValidationSandbox
)

from .hardware_fingerprint import (
    HardwareFingerprintGenerator,
    HardwareFingerprint,
    PrivacyMode,
    FingerprintingLevel,
    AntiCheatHardwareChecker
)

from .privacy_protection import (
    PrivacyProtectionSystem,
    PrivacyPolicy,
    PrivacyLevel,
    ConsentType,
    ConsentManager,
    AnonymizationResult,
    create_privacy_system_for_steam_deck
)

from .access_control import (
    AccessControlSystem,
    UserRole,
    Permission,
    ReputationAction,
    QuarantineReason,
    create_steam_deck_access_control
)

from .anticheat_compatibility import (
    AntiCheatCompatibilityChecker,
    AntiCheatSystem,
    CompatibilityLevel,
    ShaderRiskLevel,
    create_steam_deck_anticheat_checker
)

from .security_integration import (
    SecurityIntegrationLayer,
    SecurityValidationRequest,
    SecurityValidationResult,
    SecurityAction,
    create_integrated_security_system
)

from .security_testing import (
    SecurityTestSuite,
    TestCategory,
    TestResult,
    SeverityLevel,
    SecurityReport,
    run_comprehensive_security_test
)

# Convenience functions
async def create_secure_shader_system(
    enable_sandbox: bool = True,
    enable_privacy: bool = True,
    enable_anticheat_check: bool = True,
    privacy_level: PrivacyLevel = PrivacyLevel.ENHANCED,
    security_level: SecurityLevel = SecurityLevel.STRICT
):
    """
    Create a complete secure shader validation system with all components enabled.
    
    Args:
        enable_sandbox: Enable sandboxed execution for suspicious shaders
        enable_privacy: Enable privacy protection and data anonymization
        enable_anticheat_check: Enable anti-cheat compatibility checking
        privacy_level: Level of privacy protection to apply
        security_level: Sandbox security level for untrusted code
    
    Returns:
        SecurityIntegrationLayer: Complete security validation system
    """
    # Create integration layer with all security components
    security_system = await create_integrated_security_system()
    
    # Configure based on parameters
    if not enable_sandbox:
        security_system.sandbox = None
    
    if not enable_privacy:
        security_system.privacy_system = None
    
    if not enable_anticheat_check:
        security_system.anticheat_checker = None
    
    return security_system


def create_steam_deck_security_config():
    """
    Create security configuration optimized for Steam Deck deployment.
    
    Returns:
        dict: Configuration dictionary for Steam Deck optimization
    """
    return {
        'static_analysis': {
            'enable_spir_v_validation': True,
            'malware_detection': True,
            'obfuscation_detection': True,
            'performance_analysis': True
        },
        'sandbox': {
            'security_level': SecurityLevel.STRICT,
            'max_memory_mb': 256,
            'max_cpu_seconds': 30.0,
            'enable_gpu_access': True,
            'network_isolation': True
        },
        'hardware_fingerprinting': {
            'privacy_mode': PrivacyMode.SALTED,
            'steam_deck_optimized': True,
            'anticheat_compatibility': True
        },
        'privacy_protection': {
            'privacy_level': PrivacyLevel.ENHANCED,
            'differential_privacy': True,
            'gdpr_compliant': True,
            'data_retention_days': 30
        },
        'access_control': {
            'enable_reputation_system': True,
            'require_steam_verification': False,  # More lenient for Steam Deck
            'rate_limits': {
                'shader_upload': 20,  # per hour
                'shader_download': 200,  # per hour
            }
        },
        'anticheat_compatibility': {
            'enable_eac_checking': True,
            'enable_battleye_checking': True,
            'enable_vac_checking': True,
            'whitelist_steam_deck_optimizations': True
        }
    }


# Security validation shortcuts
async def validate_shader_quick(shader_data: bytes, user_id: str = None) -> dict:
    """
    Quick shader validation with basic security checks.
    
    Args:
        shader_data: SPIR-V shader bytecode
        user_id: Optional user ID for access control
    
    Returns:
        dict: Validation result with security status
    """
    analyzer = SPIRVStaticAnalyzer()
    result = analyzer.analyze_bytecode(shader_data)
    
    return {
        'is_safe': result.overall_threat_level == SecurityThreatLevel.SAFE,
        'threat_level': result.overall_threat_level.value,
        'vulnerabilities': len(result.vulnerabilities),
        'eac_compatible': result.eac_compatible,
        'battleye_compatible': result.battleye_compatible,
        'vac_compatible': result.vac_compatible,
        'analysis_time_ms': result.analysis_time_ms
    }


def check_hardware_compatibility(include_anticheat: bool = True) -> dict:
    """
    Check current hardware compatibility for shader optimization.
    
    Args:
        include_anticheat: Include anti-cheat system compatibility
    
    Returns:
        dict: Hardware compatibility information
    """
    hw_generator = HardwareFingerprintGenerator()
    fingerprint = hw_generator.generate_fingerprint()
    
    result = {
        'is_steam_deck': fingerprint.is_steam_deck,
        'hardware_id': fingerprint.fingerprint_id,
        'detection_confidence': fingerprint.detection_confidence,
        'anticheat_compatibility': {
            'eac': fingerprint.eac_compatible_hardware,
            'battleye': fingerprint.battleye_compatible_hardware,
            'vac': fingerprint.vac_compatible_hardware
        } if include_anticheat else None
    }
    
    return result


# Export main functions and classes
__all__ = [
    # Core analyzers
    'SPIRVStaticAnalyzer',
    'ShaderSandbox', 
    'HardwareFingerprintGenerator',
    'PrivacyProtectionSystem',
    'AccessControlSystem',
    'AntiCheatCompatibilityChecker',
    'SecurityIntegrationLayer',
    'SecurityTestSuite',
    
    # Enums and types
    'SecurityThreatLevel',
    'SecurityLevel',
    'PrivacyMode',
    'PrivacyLevel',
    'UserRole',
    'Permission',
    'AntiCheatSystem',
    'CompatibilityLevel',
    'SecurityAction',
    'TestCategory',
    'TestResult',
    
    # Result types
    'SPIRVAnalysisResult',
    'ExecutionResult',
    'HardwareFingerprint',
    'AnonymizationResult',
    'SecurityValidationResult',
    'SecurityReport',
    
    # Convenience functions
    'create_secure_shader_system',
    'create_steam_deck_security_config',
    'validate_shader_quick',
    'check_hardware_compatibility',
    
    # Factory functions
    'create_test_analyzer',
    'create_privacy_system_for_steam_deck',
    'create_steam_deck_access_control',
    'create_steam_deck_anticheat_checker',
    'create_integrated_security_system',
    'run_comprehensive_security_test',
]