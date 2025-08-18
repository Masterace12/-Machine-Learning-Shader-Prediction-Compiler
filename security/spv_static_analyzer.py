#!/usr/bin/env python3
"""
SPIR-V Bytecode Static Analysis Engine for Shader Security Validation

This module provides comprehensive static analysis of SPIR-V bytecode to detect:
- Malicious patterns and obfuscation techniques
- Resource access violations and potential exploits
- Code injection attempts and suspicious constructions
- Compliance violations with Vulkan/DirectX specifications
- Known vulnerability signatures and exploit patterns

Security Classifications:
- SAFE: Shader passes all security checks
- SUSPICIOUS: Shader has concerning patterns but may be legitimate
- MALICIOUS: Shader contains definitive malicious patterns
- INVALID: Shader violates specifications or is corrupted
"""

import struct
import hashlib
import logging
import time
import re
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
import json
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class SecurityThreatLevel(Enum):
    """Security threat classification levels"""
    SAFE = "safe"
    SUSPICIOUS = "suspicious" 
    MALICIOUS = "malicious"
    INVALID = "invalid"


class SPIRVOpcode(IntEnum):
    """SPIR-V Opcodes relevant for security analysis"""
    # Memory operations - potential attack vectors
    OpLoad = 61
    OpStore = 62
    OpImageRead = 98
    OpImageWrite = 99
    OpAtomicLoad = 227
    OpAtomicStore = 228
    OpAtomicExchange = 229
    OpAtomicCompareExchange = 230
    OpAtomicIIncrement = 231
    OpAtomicIDecrement = 232
    OpAtomicIAdd = 233
    OpAtomicISub = 234
    OpAtomicSMin = 235
    OpAtomicUMin = 236
    OpAtomicSMax = 237
    OpAtomicUMax = 238
    OpAtomicAnd = 239
    OpAtomicOr = 240
    OpAtomicXor = 241
    
    # Control flow - obfuscation detection
    OpBranch = 249
    OpBranchConditional = 250
    OpSwitch = 251
    OpKill = 252
    OpReturn = 253
    OpReturnValue = 254
    
    # Function calls - potential injection points  
    OpFunctionCall = 57
    OpFunctionParameter = 55
    OpFunction = 54
    OpFunctionEnd = 56
    
    # Extensions - potential attack vectors
    OpExtInstImport = 11
    OpExtInst = 12
    
    # Debug info that could leak information
    OpName = 5
    OpMemberName = 6
    OpString = 7
    OpLine = 8
    OpNoLine = 317
    
    # Decorations
    OpDecorate = 71
    OpMemberDecorate = 72
    OpDecorationGroup = 73


@dataclass
class SecurityVulnerability:
    """Detected security vulnerability"""
    threat_level: SecurityThreatLevel
    category: str
    description: str
    location_offset: int
    evidence: str
    mitigation: str
    confidence: float  # 0.0 to 1.0


@dataclass 
class SPIRVAnalysisResult:
    """Complete SPIR-V security analysis result"""
    shader_hash: str
    file_size: int
    analysis_time_ms: float
    overall_threat_level: SecurityThreatLevel
    vulnerabilities: List[SecurityVulnerability]
    
    # SPIR-V structure validation
    is_valid_spirv: bool
    version: Tuple[int, int]
    generator_id: int
    bound: int
    
    # Security metrics
    instruction_count: int
    function_count: int
    memory_operations: int
    atomic_operations: int
    branch_complexity: int
    extension_usage: List[str]
    
    # Obfuscation detection
    obfuscation_score: float  # 0.0 to 1.0
    entropy_score: float
    suspicious_patterns: List[str]
    
    # Resource usage analysis  
    estimated_memory_usage: int
    register_pressure: float
    execution_complexity: float
    
    # Anti-cheat compatibility
    eac_compatible: bool
    battleye_compatible: bool
    vac_compatible: bool
    
    # Metadata safety
    contains_pii: bool
    debug_info_stripped: bool
    source_references: List[str]


class MalwareSignatureDatabase:
    """Database of known malicious shader patterns"""
    
    def __init__(self):
        self.signatures = self._load_builtin_signatures()
        self.custom_signatures = {}
        
    def _load_builtin_signatures(self) -> Dict[str, Dict]:
        """Load built-in malware signatures"""
        return {
            # Buffer overflow attempts
            "buffer_overflow_1": {
                "pattern": [SPIRVOpcode.OpLoad, SPIRVOpcode.OpStore, SPIRVOpcode.OpLoad],
                "threat_level": SecurityThreatLevel.MALICIOUS,
                "description": "Potential buffer overflow pattern",
                "confidence": 0.8
            },
            
            # Excessive atomic operations (DoS attempt)  
            "atomic_dos": {
                "pattern": "atomic_ops_threshold",
                "threshold": 100,
                "threat_level": SecurityThreatLevel.SUSPICIOUS,
                "description": "Excessive atomic operations - potential DoS",
                "confidence": 0.7
            },
            
            # Complex control flow obfuscation
            "control_flow_obfuscation": {
                "pattern": "complex_branches",
                "threshold": 50,
                "threat_level": SecurityThreatLevel.SUSPICIOUS, 
                "description": "Highly obfuscated control flow",
                "confidence": 0.6
            },
            
            # Suspicious extension usage
            "unknown_extension": {
                "pattern": "extension_whitelist",
                "threat_level": SecurityThreatLevel.SUSPICIOUS,
                "description": "Usage of non-standard extensions",
                "confidence": 0.5
            },
            
            # Information leakage patterns
            "info_leak": {
                "pattern": [SPIRVOpcode.OpString],
                "contains": ["password", "token", "key", "secret"],
                "threat_level": SecurityThreatLevel.SUSPICIOUS,
                "description": "Potential information leakage in strings",
                "confidence": 0.9
            }
        }
    
    def add_signature(self, name: str, signature: Dict):
        """Add custom malware signature"""
        self.custom_signatures[name] = signature
        logger.info(f"Added custom security signature: {name}")
    
    def get_all_signatures(self) -> Dict[str, Dict]:
        """Get all signatures (builtin + custom)"""
        return {**self.signatures, **self.custom_signatures}


class SPIRVStaticAnalyzer:
    """Static analysis engine for SPIR-V bytecode security validation"""
    
    def __init__(self):
        self.signature_db = MalwareSignatureDatabase()
        self.whitelisted_extensions = {
            "GLSL.std.450", "SPV_KHR_storage_buffer_storage_class",
            "SPV_KHR_variable_pointers", "SPV_EXT_shader_stencil_export",
            "SPV_AMD_shader_trinary_minmax", "SPV_NV_geometry_shader_passthrough"
        }
        self.anti_cheat_incompatible_ops = {
            SPIRVOpcode.OpAtomicLoad, SPIRVOpcode.OpAtomicStore,
            SPIRVOpcode.OpImageWrite  # Write operations are often flagged
        }
        
        logger.info("SPIR-V static analyzer initialized")
    
    def analyze_bytecode(self, bytecode: bytes, shader_hash: str = None) -> SPIRVAnalysisResult:
        """Perform comprehensive static analysis of SPIR-V bytecode"""
        start_time = time.time()
        
        if shader_hash is None:
            shader_hash = hashlib.sha256(bytecode).hexdigest()[:16]
        
        logger.debug(f"Starting static analysis of shader {shader_hash}")
        
        # Initialize result
        result = SPIRVAnalysisResult(
            shader_hash=shader_hash,
            file_size=len(bytecode),
            analysis_time_ms=0,
            overall_threat_level=SecurityThreatLevel.SAFE,
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
            eac_compatible=True,
            battleye_compatible=True,
            vac_compatible=True,
            contains_pii=False,
            debug_info_stripped=True,
            source_references=[]
        )
        
        try:
            # 1. Validate SPIR-V structure
            if not self._validate_spirv_header(bytecode, result):
                result.overall_threat_level = SecurityThreatLevel.INVALID
                return result
            
            # 2. Parse instructions
            instructions = self._parse_instructions(bytecode, result)
            if not instructions:
                result.overall_threat_level = SecurityThreatLevel.INVALID
                return result
            
            # 3. Analyze instruction patterns
            self._analyze_instruction_patterns(instructions, result)
            
            # 4. Detect malware signatures
            self._detect_malware_signatures(instructions, result)
            
            # 5. Analyze obfuscation techniques
            self._analyze_obfuscation(instructions, bytecode, result)
            
            # 6. Check resource usage
            self._analyze_resource_usage(instructions, result)
            
            # 7. Anti-cheat compatibility check
            self._check_anticheat_compatibility(instructions, result)
            
            # 8. Privacy analysis
            self._analyze_privacy_concerns(instructions, bytecode, result)
            
            # 9. Calculate overall threat level
            result.overall_threat_level = self._calculate_overall_threat_level(result)
            
            logger.info(f"Analysis complete for {shader_hash}: {result.overall_threat_level.value} "
                       f"({len(result.vulnerabilities)} vulnerabilities)")
            
        except Exception as e:
            logger.error(f"Analysis error for {shader_hash}: {e}")
            result.overall_threat_level = SecurityThreatLevel.INVALID
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.INVALID,
                category="analysis_error",
                description=f"Analysis failed: {str(e)}",
                location_offset=0,
                evidence="",
                mitigation="Manual review required",
                confidence=1.0
            ))
        
        finally:
            result.analysis_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _validate_spirv_header(self, bytecode: bytes, result: SPIRVAnalysisResult) -> bool:
        """Validate SPIR-V header structure"""
        if len(bytecode) < 20:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.INVALID,
                category="header_validation",
                description="File too small to be valid SPIR-V",
                location_offset=0,
                evidence=f"File size: {len(bytecode)} bytes",
                mitigation="Reject file",
                confidence=1.0
            ))
            return False
        
        # Check magic number
        magic = struct.unpack('<I', bytecode[0:4])[0]
        if magic != 0x07230203:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.INVALID,
                category="header_validation", 
                description="Invalid SPIR-V magic number",
                location_offset=0,
                evidence=f"Magic: 0x{magic:08x}",
                mitigation="Reject file",
                confidence=1.0
            ))
            return False
        
        # Parse header
        try:
            version = struct.unpack('<I', bytecode[4:8])[0]
            result.version = ((version >> 16) & 0xFF, (version >> 8) & 0xFF)
            result.generator_id = struct.unpack('<I', bytecode[8:12])[0] 
            result.bound = struct.unpack('<I', bytecode[12:16])[0]
            schema = struct.unpack('<I', bytecode[16:20])[0]
            
            # Validate version
            if result.version[0] > 1 or (result.version[0] == 1 and result.version[1] > 6):
                result.vulnerabilities.append(SecurityVulnerability(
                    threat_level=SecurityThreatLevel.SUSPICIOUS,
                    category="header_validation",
                    description="Unsupported SPIR-V version",
                    location_offset=4,
                    evidence=f"Version: {result.version[0]}.{result.version[1]}",
                    mitigation="Verify compatibility",
                    confidence=0.8
                ))
            
            # Validate bound
            if result.bound > 1000000:  # Suspiciously high ID bound
                result.vulnerabilities.append(SecurityVulnerability(
                    threat_level=SecurityThreatLevel.SUSPICIOUS,
                    category="resource_usage",
                    description="Suspiciously high ID bound",
                    location_offset=12,
                    evidence=f"Bound: {result.bound}",
                    mitigation="Monitor memory usage",
                    confidence=0.7
                ))
            
            # Check schema (must be 0)
            if schema != 0:
                result.vulnerabilities.append(SecurityVulnerability(
                    threat_level=SecurityThreatLevel.INVALID,
                    category="header_validation",
                    description="Invalid SPIR-V schema",
                    location_offset=16, 
                    evidence=f"Schema: {schema}",
                    mitigation="Reject file",
                    confidence=1.0
                ))
                return False
            
            result.is_valid_spirv = True
            return True
            
        except struct.error as e:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.INVALID,
                category="header_validation",
                description=f"Header parsing error: {e}",
                location_offset=0,
                evidence="",
                mitigation="Reject file",
                confidence=1.0
            ))
            return False
    
    def _parse_instructions(self, bytecode: bytes, result: SPIRVAnalysisResult) -> List[Dict]:
        """Parse SPIR-V instructions"""
        instructions = []
        offset = 20  # Skip header
        
        try:
            while offset < len(bytecode):
                if offset + 4 > len(bytecode):
                    break
                
                # Read instruction header
                word = struct.unpack('<I', bytecode[offset:offset+4])[0]
                opcode = word & 0xFFFF
                length = word >> 16
                
                if length == 0:
                    result.vulnerabilities.append(SecurityVulnerability(
                        threat_level=SecurityThreatLevel.INVALID,
                        category="instruction_parsing",
                        description="Zero-length instruction",
                        location_offset=offset,
                        evidence=f"Word: 0x{word:08x}",
                        mitigation="Reject file",
                        confidence=1.0
                    ))
                    break
                
                # Check bounds
                instruction_end = offset + (length * 4)
                if instruction_end > len(bytecode):
                    result.vulnerabilities.append(SecurityVulnerability(
                        threat_level=SecurityThreatLevel.INVALID,
                        category="instruction_parsing",
                        description="Instruction extends beyond file",
                        location_offset=offset,
                        evidence=f"Length: {length}, remaining: {len(bytecode) - offset}",
                        mitigation="Reject file", 
                        confidence=1.0
                    ))
                    break
                
                # Parse operands
                operands = []
                for i in range(1, length):
                    operand_offset = offset + (i * 4)
                    if operand_offset + 4 <= len(bytecode):
                        operand = struct.unpack('<I', bytecode[operand_offset:operand_offset+4])[0]
                        operands.append(operand)
                
                instruction = {
                    'offset': offset,
                    'opcode': opcode,
                    'length': length,
                    'operands': operands
                }
                instructions.append(instruction)
                
                offset = instruction_end
                result.instruction_count += 1
            
            return instructions
            
        except Exception as e:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.INVALID,
                category="instruction_parsing",
                description=f"Instruction parsing error: {e}",
                location_offset=offset,
                evidence="",
                mitigation="Reject file",
                confidence=1.0
            ))
            return []
    
    def _analyze_instruction_patterns(self, instructions: List[Dict], result: SPIRVAnalysisResult):
        """Analyze instruction patterns for security issues"""
        
        function_depth = 0
        branch_depth = 0
        max_branch_depth = 0
        memory_ops = 0
        atomic_ops = 0
        
        for instruction in instructions:
            opcode = instruction['opcode']
            
            # Count functions
            if opcode == SPIRVOpcode.OpFunction:
                function_depth += 1
                result.function_count += 1
            elif opcode == SPIRVOpcode.OpFunctionEnd:
                function_depth -= 1
            
            # Track branch complexity  
            if opcode in [SPIRVOpcode.OpBranch, SPIRVOpcode.OpBranchConditional, SPIRVOpcode.OpSwitch]:
                branch_depth += 1
                max_branch_depth = max(max_branch_depth, branch_depth)
                result.branch_complexity += 1
            elif opcode in [SPIRVOpcode.OpReturn, SPIRVOpcode.OpReturnValue]:
                branch_depth = max(0, branch_depth - 1)
            
            # Count memory operations
            if opcode in [SPIRVOpcode.OpLoad, SPIRVOpcode.OpStore, 
                         SPIRVOpcode.OpImageRead, SPIRVOpcode.OpImageWrite]:
                memory_ops += 1
            
            # Count atomic operations
            if opcode >= SPIRVOpcode.OpAtomicLoad and opcode <= SPIRVOpcode.OpAtomicXor:
                atomic_ops += 1
            
            # Check for extensions
            if opcode == SPIRVOpcode.OpExtInstImport:
                # Would need to parse string to get extension name
                result.extension_usage.append("unknown_extension")
        
        result.memory_operations = memory_ops
        result.atomic_operations = atomic_ops
        
        # Validate function nesting
        if function_depth != 0:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.INVALID,
                category="structure_validation",
                description="Unbalanced function declarations",
                location_offset=0,
                evidence=f"Function depth: {function_depth}",
                mitigation="Reject file",
                confidence=1.0
            ))
        
        # Check for excessive branching (obfuscation indicator)
        if max_branch_depth > 20:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="obfuscation",
                description="Excessive branch nesting depth",
                location_offset=0,
                evidence=f"Max depth: {max_branch_depth}",
                mitigation="Manual review required",
                confidence=0.8
            ))
    
    def _detect_malware_signatures(self, instructions: List[Dict], result: SPIRVAnalysisResult):
        """Detect known malware signatures in instruction patterns"""
        
        signatures = self.signature_db.get_all_signatures()
        
        for sig_name, sig_data in signatures.items():
            pattern = sig_data.get('pattern')
            threat_level = sig_data.get('threat_level')
            description = sig_data.get('description')
            confidence = sig_data.get('confidence', 0.5)
            
            detected = False
            evidence = ""
            
            if isinstance(pattern, list):
                # Pattern matching
                detected, evidence = self._match_instruction_pattern(instructions, pattern)
            elif pattern == "atomic_ops_threshold":
                threshold = sig_data.get('threshold', 100)
                if result.atomic_operations > threshold:
                    detected = True
                    evidence = f"Atomic operations: {result.atomic_operations}"
            elif pattern == "complex_branches":
                threshold = sig_data.get('threshold', 50)
                if result.branch_complexity > threshold:
                    detected = True
                    evidence = f"Branch complexity: {result.branch_complexity}"
            elif pattern == "extension_whitelist":
                for ext in result.extension_usage:
                    if ext not in self.whitelisted_extensions:
                        detected = True
                        evidence = f"Non-whitelisted extension: {ext}"
                        break
            
            if detected:
                result.vulnerabilities.append(SecurityVulnerability(
                    threat_level=threat_level,
                    category="malware_signature",
                    description=description,
                    location_offset=0,
                    evidence=evidence,
                    mitigation="Block or quarantine shader",
                    confidence=confidence
                ))
                
                result.suspicious_patterns.append(sig_name)
    
    def _match_instruction_pattern(self, instructions: List[Dict], pattern: List[int]) -> Tuple[bool, str]:
        """Match a specific instruction pattern"""
        if len(pattern) > len(instructions):
            return False, ""
        
        for i in range(len(instructions) - len(pattern) + 1):
            match = True
            for j, expected_opcode in enumerate(pattern):
                if instructions[i + j]['opcode'] != expected_opcode:
                    match = False
                    break
            
            if match:
                return True, f"Pattern matched at offset {instructions[i]['offset']}"
        
        return False, ""
    
    def _analyze_obfuscation(self, instructions: List[Dict], bytecode: bytes, result: SPIRVAnalysisResult):
        """Analyze for obfuscation techniques"""
        
        # Calculate entropy of bytecode
        byte_counts = Counter(bytecode)
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / len(bytecode)
            entropy -= probability * (probability.log2() if probability > 0 else 0)
        
        result.entropy_score = entropy / 8.0  # Normalize to 0-1
        
        # High entropy indicates potential obfuscation
        if result.entropy_score > 0.85:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="obfuscation",
                description="High bytecode entropy - possible obfuscation",
                location_offset=0,
                evidence=f"Entropy: {result.entropy_score:.3f}",
                mitigation="Deep analysis required",
                confidence=0.7
            ))
        
        # Check for unusual instruction distribution
        opcode_counts = Counter(inst['opcode'] for inst in instructions)
        
        # Dead code detection (unreachable instructions)
        reachable = self._analyze_reachability(instructions)
        unreachable_count = len(instructions) - len(reachable)
        
        if unreachable_count > len(instructions) * 0.1:  # >10% unreachable
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="obfuscation",
                description="Significant amount of dead code",
                location_offset=0,
                evidence=f"Unreachable instructions: {unreachable_count}/{len(instructions)}",
                mitigation="Remove dead code",
                confidence=0.6
            ))
        
        # Calculate obfuscation score based on multiple factors
        factors = [
            result.entropy_score,
            min(1.0, result.branch_complexity / 100.0),
            min(1.0, unreachable_count / len(instructions)) if instructions else 0,
            min(1.0, len(result.suspicious_patterns) / 5.0)
        ]
        
        result.obfuscation_score = sum(factors) / len(factors)
    
    def _analyze_reachability(self, instructions: List[Dict]) -> Set[int]:
        """Analyze instruction reachability for dead code detection"""
        reachable = {0}  # Start from first instruction
        
        # Simple reachability analysis
        for i, instruction in enumerate(instructions):
            if i not in reachable:
                continue
            
            opcode = instruction['opcode']
            
            # Sequential execution
            if opcode not in [SPIRVOpcode.OpBranch, SPIRVOpcode.OpReturn, 
                             SPIRVOpcode.OpReturnValue, SPIRVOpcode.OpKill]:
                if i + 1 < len(instructions):
                    reachable.add(i + 1)
            
            # Branch targets (simplified - would need proper parsing)
            if opcode in [SPIRVOpcode.OpBranch, SPIRVOpcode.OpBranchConditional]:
                # Would need to parse branch targets from operands
                pass
        
        return reachable
    
    def _analyze_resource_usage(self, instructions: List[Dict], result: SPIRVAnalysisResult):
        """Analyze resource usage patterns"""
        
        # Estimate memory usage
        result.estimated_memory_usage = result.bound * 4  # Rough estimate
        
        # Calculate register pressure
        result.register_pressure = min(1.0, result.bound / 1000.0)  # Normalize
        
        # Calculate execution complexity
        complexity_factors = [
            result.instruction_count / 1000.0,
            result.branch_complexity / 100.0,
            result.memory_operations / 100.0,
            result.atomic_operations / 50.0
        ]
        result.execution_complexity = min(1.0, sum(complexity_factors) / len(complexity_factors))
        
        # Check for resource exhaustion attacks
        if result.estimated_memory_usage > 100 * 1024 * 1024:  # >100MB
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="resource_usage",
                description="Excessive memory usage",
                location_offset=0,
                evidence=f"Estimated usage: {result.estimated_memory_usage // (1024*1024)}MB",
                mitigation="Resource limits required",
                confidence=0.8
            ))
        
        if result.instruction_count > 10000:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="resource_usage",
                description="Excessive instruction count",
                location_offset=0,
                evidence=f"Instructions: {result.instruction_count}",
                mitigation="Execution time limits required",
                confidence=0.7
            ))
    
    def _check_anticheat_compatibility(self, instructions: List[Dict], result: SPIRVAnalysisResult):
        """Check compatibility with anti-cheat systems"""
        
        incompatible_ops = []
        
        for instruction in instructions:
            opcode = instruction['opcode']
            
            if opcode in self.anti_cheat_incompatible_ops:
                incompatible_ops.append(opcode)
        
        # EAC is strict about atomic operations
        if any(op >= SPIRVOpcode.OpAtomicLoad and op <= SPIRVOpcode.OpAtomicXor 
               for op in incompatible_ops):
            result.eac_compatible = False
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="anticheat_compatibility", 
                description="Contains operations flagged by EAC",
                location_offset=0,
                evidence=f"Atomic operations: {result.atomic_operations}",
                mitigation="Remove atomic operations or whitelist",
                confidence=0.9
            ))
        
        # BattlEye flags image write operations
        if SPIRVOpcode.OpImageWrite in incompatible_ops:
            result.battleye_compatible = False
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="anticheat_compatibility",
                description="Contains operations flagged by BattlEye", 
                location_offset=0,
                evidence="Image write operations detected",
                mitigation="Remove image write operations",
                confidence=0.8
            ))
        
        # VAC is generally more lenient but flags obvious cheating patterns
        if result.obfuscation_score > 0.8:
            result.vac_compatible = False
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="anticheat_compatibility",
                description="High obfuscation may trigger VAC",
                location_offset=0,
                evidence=f"Obfuscation score: {result.obfuscation_score:.3f}",
                mitigation="Reduce obfuscation",
                confidence=0.6
            ))
    
    def _analyze_privacy_concerns(self, instructions: List[Dict], bytecode: bytes, result: SPIRVAnalysisResult):
        """Analyze for privacy and information leakage concerns"""
        
        # Check for debug information
        debug_ops = [SPIRVOpcode.OpName, SPIRVOpcode.OpMemberName, 
                    SPIRVOpcode.OpString, SPIRVOpcode.OpLine]
        
        has_debug_info = any(inst['opcode'] in debug_ops for inst in instructions)
        result.debug_info_stripped = not has_debug_info
        
        if has_debug_info:
            result.vulnerabilities.append(SecurityVulnerability(
                threat_level=SecurityThreatLevel.SUSPICIOUS,
                category="privacy",
                description="Contains debug information",
                location_offset=0,
                evidence="Debug symbols present",
                mitigation="Strip debug information",
                confidence=0.5
            ))
        
        # Check for embedded strings that might contain PII
        pii_patterns = [
            rb'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
            rb'\b\d{3}-\d{2}-\d{4}\b',  # SSN format
            rb'\b\d{16}\b',  # Credit card format
            rb'[Pp]assword',  # Password references
            rb'[Tt]oken',     # Token references
            rb'[Kk]ey',       # Key references
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, bytecode):
                result.contains_pii = True
                result.vulnerabilities.append(SecurityVulnerability(
                    threat_level=SecurityThreatLevel.SUSPICIOUS,
                    category="privacy",
                    description="Potential PII detected in shader data",
                    location_offset=0,
                    evidence="Sensitive pattern matched",
                    mitigation="Remove sensitive information",
                    confidence=0.7
                ))
                break
        
        # Check for source file references
        source_patterns = [rb'\.glsl', rb'\.hlsl', rb'\.frag', rb'\.vert', rb'\.comp']
        
        for pattern in source_patterns:
            if re.search(pattern, bytecode, re.IGNORECASE):
                result.source_references.append(pattern.decode('utf-8', errors='ignore'))
    
    def _calculate_overall_threat_level(self, result: SPIRVAnalysisResult) -> SecurityThreatLevel:
        """Calculate overall threat level based on all analysis results"""
        
        if not result.is_valid_spirv:
            return SecurityThreatLevel.INVALID
        
        # Count vulnerabilities by severity
        malicious_count = sum(1 for v in result.vulnerabilities 
                             if v.threat_level == SecurityThreatLevel.MALICIOUS)
        suspicious_count = sum(1 for v in result.vulnerabilities 
                              if v.threat_level == SecurityThreatLevel.SUSPICIOUS)
        invalid_count = sum(1 for v in result.vulnerabilities 
                           if v.threat_level == SecurityThreatLevel.INVALID)
        
        # Any invalid or malicious findings = overall threat
        if invalid_count > 0:
            return SecurityThreatLevel.INVALID
        if malicious_count > 0:
            return SecurityThreatLevel.MALICIOUS
        
        # Multiple suspicious findings = elevated threat
        if suspicious_count >= 3:
            return SecurityThreatLevel.MALICIOUS
        elif suspicious_count >= 1:
            return SecurityThreatLevel.SUSPICIOUS
        
        # Check composite scores
        if (result.obfuscation_score > 0.8 or 
            result.execution_complexity > 0.9 or
            result.entropy_score > 0.9):
            return SecurityThreatLevel.SUSPICIOUS
        
        return SecurityThreatLevel.SAFE
    
    def generate_security_report(self, result: SPIRVAnalysisResult) -> str:
        """Generate human-readable security analysis report"""
        
        report = [
            f"SPIR-V Security Analysis Report",
            f"=" * 40,
            f"Shader Hash: {result.shader_hash}",
            f"File Size: {result.file_size:,} bytes",
            f"Analysis Time: {result.analysis_time_ms:.1f} ms",
            f"Overall Threat Level: {result.overall_threat_level.value.upper()}",
            "",
            f"SPIR-V Structure:",
            f"  Valid: {result.is_valid_spirv}",
            f"  Version: {result.version[0]}.{result.version[1]}",
            f"  Generator ID: 0x{result.generator_id:08x}",
            f"  ID Bound: {result.bound:,}",
            "",
            f"Code Analysis:",
            f"  Instructions: {result.instruction_count:,}",
            f"  Functions: {result.function_count}",
            f"  Memory Operations: {result.memory_operations}",
            f"  Atomic Operations: {result.atomic_operations}",
            f"  Branch Complexity: {result.branch_complexity}",
            "",
            f"Security Metrics:",
            f"  Obfuscation Score: {result.obfuscation_score:.3f}",
            f"  Entropy Score: {result.entropy_score:.3f}",
            f"  Execution Complexity: {result.execution_complexity:.3f}",
            "",
            f"Anti-Cheat Compatibility:",
            f"  EAC Compatible: {result.eac_compatible}",
            f"  BattlEye Compatible: {result.battleye_compatible}",
            f"  VAC Compatible: {result.vac_compatible}",
            "",
            f"Privacy Analysis:",
            f"  Contains PII: {result.contains_pii}",
            f"  Debug Info Stripped: {result.debug_info_stripped}",
            "",
        ]
        
        if result.vulnerabilities:
            report.extend([
                f"Vulnerabilities Found ({len(result.vulnerabilities)}):",
                f"=" * 30
            ])
            
            for i, vuln in enumerate(result.vulnerabilities, 1):
                report.extend([
                    f"{i}. {vuln.category.upper()}: {vuln.description}",
                    f"   Threat Level: {vuln.threat_level.value}",
                    f"   Confidence: {vuln.confidence:.1%}",
                    f"   Location: 0x{vuln.location_offset:x}",
                    f"   Evidence: {vuln.evidence}",
                    f"   Mitigation: {vuln.mitigation}",
                    ""
                ])
        else:
            report.append("No vulnerabilities detected.")
        
        return "\n".join(report)
    
    def export_analysis_results(self, results: List[SPIRVAnalysisResult], output_path: Path):
        """Export analysis results to JSON for further processing"""
        
        export_data = {
            'export_timestamp': time.time(),
            'analyzer_version': '1.0',
            'results_count': len(results),
            'results': []
        }
        
        for result in results:
            result_data = {
                'shader_hash': result.shader_hash,
                'overall_threat_level': result.overall_threat_level.value,
                'analysis_time_ms': result.analysis_time_ms,
                'vulnerabilities': [
                    {
                        'threat_level': v.threat_level.value,
                        'category': v.category,
                        'description': v.description,
                        'confidence': v.confidence
                    }
                    for v in result.vulnerabilities
                ],
                'metrics': {
                    'instruction_count': result.instruction_count,
                    'obfuscation_score': result.obfuscation_score,
                    'entropy_score': result.entropy_score,
                    'execution_complexity': result.execution_complexity
                },
                'compatibility': {
                    'eac_compatible': result.eac_compatible,
                    'battleye_compatible': result.battleye_compatible,
                    'vac_compatible': result.vac_compatible
                }
            }
            export_data['results'].append(result_data)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(results)} analysis results to {output_path}")


def create_test_analyzer() -> SPIRVStaticAnalyzer:
    """Create analyzer instance for testing"""
    analyzer = SPIRVStaticAnalyzer()
    
    # Add some custom signatures for demonstration
    analyzer.signature_db.add_signature("crypto_mining", {
        "pattern": "atomic_ops_threshold",
        "threshold": 200,
        "threat_level": SecurityThreatLevel.MALICIOUS,
        "description": "Potential cryptocurrency mining shader",
        "confidence": 0.9
    })
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    analyzer = create_test_analyzer()
    
    # Test with sample bytecode (would be real SPIR-V in practice)
    sample_spirv = bytes([
        0x03, 0x02, 0x23, 0x07,  # Magic number
        0x00, 0x01, 0x06, 0x00,  # Version 1.6
        0x00, 0x00, 0x00, 0x00,  # Generator
        0x10, 0x00, 0x00, 0x00,  # Bound
        0x00, 0x00, 0x00, 0x00,  # Schema
        # Some fake instructions...
        0x0E, 0x00, 0x03, 0x00,  # OpMemoryModel
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    ])
    
    result = analyzer.analyze_bytecode(sample_spirv)
    
    print(analyzer.generate_security_report(result))
    print(f"\nThreat Level: {result.overall_threat_level.value}")