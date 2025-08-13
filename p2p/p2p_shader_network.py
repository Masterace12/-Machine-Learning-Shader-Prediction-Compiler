"""
Peer-to-Peer Shader Cache Distribution Network for Steam Deck
Implements secure, efficient shader cache sharing across Steam Deck users
"""

import asyncio
import hashlib
import json
import time
import random
import struct
import socket
import threading
from typing import Dict, List, Set, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import logging
import zlib
import lzma
from enum import Enum
import pickle
import ipaddress
import numpy as np

# Cryptographic imports
try:
    import cryptography
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: Cryptography not available, using basic hashing only")

# Import existing shader prediction system
from shader_prediction_system import ShaderMetrics, ShaderType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeerRole(Enum):
    """Roles in the P2P network"""
    SEEDER = "seeder"      # High-capacity peer with good connection
    LEECHER = "leecher"    # Standard Steam Deck peer
    BOOTSTRAP = "bootstrap" # Network bootstrap node
    RELAY = "relay"        # NAT traversal relay node


class MessageType(Enum):
    """P2P message types"""
    PING = "ping"
    PONG = "pong"
    PEER_REQUEST = "peer_request"
    PEER_RESPONSE = "peer_response"
    CACHE_REQUEST = "cache_request"
    CACHE_RESPONSE = "cache_response"
    CACHE_ANNOUNCEMENT = "cache_announcement"
    REPUTATION_UPDATE = "reputation_update"
    DHT_STORE = "dht_store"
    DHT_FIND = "dht_find"
    DHT_RESPONSE = "dht_response"
    HEARTBEAT = "heartbeat"
    NAT_TRAVERSAL = "nat_traversal"


class CompressionMethod(Enum):
    """Compression methods for shader data"""
    NONE = "none"
    ZLIB = "zlib"
    LZMA = "lzma"
    CUSTOM_DELTA = "custom_delta"


@dataclass
class PeerInfo:
    """Information about a peer in the network"""
    peer_id: str
    address: str
    port: int
    role: PeerRole
    public_key: Optional[bytes] = None
    last_seen: float = field(default_factory=time.time)
    reputation_score: float = 0.0
    bandwidth_capacity: int = 1024  # KB/s
    cache_size: int = 0  # Number of cached shaders
    uptime_hours: float = 0.0
    nat_type: str = "unknown"
    steam_deck_verified: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'peer_id': self.peer_id,
            'address': self.address,
            'port': self.port,
            'role': self.role.value,
            'last_seen': self.last_seen,
            'reputation_score': self.reputation_score,
            'bandwidth_capacity': self.bandwidth_capacity,
            'cache_size': self.cache_size,
            'uptime_hours': self.uptime_hours,
            'nat_type': self.nat_type,
            'steam_deck_verified': self.steam_deck_verified
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PeerInfo':
        return cls(
            peer_id=data['peer_id'],
            address=data['address'],
            port=data['port'],
            role=PeerRole(data['role']),
            last_seen=data.get('last_seen', time.time()),
            reputation_score=data.get('reputation_score', 0.0),
            bandwidth_capacity=data.get('bandwidth_capacity', 1024),
            cache_size=data.get('cache_size', 0),
            uptime_hours=data.get('uptime_hours', 0.0),
            nat_type=data.get('nat_type', 'unknown'),
            steam_deck_verified=data.get('steam_deck_verified', False)
        )


@dataclass
class ShaderCacheEntry:
    """A cached shader with metadata"""
    shader_hash: str
    game_id: str
    shader_type: ShaderType
    compressed_data: bytes
    compression_method: CompressionMethod
    original_size: int
    compressed_size: int
    checksum: str
    creation_time: float
    download_count: int = 0
    validation_count: int = 0
    validation_failures: int = 0
    source_peer: Optional[str] = None
    signature: Optional[bytes] = None
    
    def to_dict(self) -> Dict:
        return {
            'shader_hash': self.shader_hash,
            'game_id': self.game_id,
            'shader_type': self.shader_type.value,
            'compressed_size': self.compressed_size,
            'original_size': self.original_size,
            'compression_method': self.compression_method.value,
            'checksum': self.checksum,
            'creation_time': self.creation_time,
            'download_count': self.download_count,
            'validation_count': self.validation_count,
            'validation_failures': self.validation_failures,
            'source_peer': self.source_peer
        }
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio"""
        if self.original_size == 0:
            return 0.0
        return 1.0 - (self.compressed_size / self.original_size)


@dataclass
class P2PMessage:
    """Standard P2P network message"""
    msg_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    payload: Dict
    timestamp: float = field(default_factory=time.time)
    sequence_id: int = 0
    signature: Optional[bytes] = None
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        msg_dict = {
            'type': self.msg_type.value,
            'sender': self.sender_id,
            'recipient': self.recipient_id,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'sequence_id': self.sequence_id
        }
        return json.dumps(msg_dict).encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'P2PMessage':
        """Deserialize message from bytes"""
        msg_dict = json.loads(data.decode('utf-8'))
        return cls(
            msg_type=MessageType(msg_dict['type']),
            sender_id=msg_dict['sender'],
            recipient_id=msg_dict.get('recipient'),
            payload=msg_dict['payload'],
            timestamp=msg_dict['timestamp'],
            sequence_id=msg_dict.get('sequence_id', 0)
        )


class CryptographicValidator:
    """Handles cryptographic validation of shader caches"""
    
    def __init__(self):
        self.key_pairs = {}  # peer_id -> (public_key, private_key)
        self.trusted_keys = set()  # Set of trusted public key fingerprints
        
    def generate_key_pair(self, peer_id: str) -> Tuple[bytes, bytes]:
        """Generate RSA key pair for peer"""
        if not CRYPTO_AVAILABLE:
            # CRITICAL: Crypto libraries are required for security
            raise RuntimeError("Cryptography libraries are required for P2P network security. "
                              "Install with: pip install cryptography>=37.0.0")
            
        # Generate RSA key pair with enhanced security
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096  # Enhanced security - 4096-bit keys
        )
        public_key = private_key.public_key()
        
        # Create secure password for private key encryption
        key_password = os.urandom(32)  # 256-bit random password
        
        # Serialize keys with encryption
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(key_password)
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        # Store with encrypted private key and password
        self.key_pairs[peer_id] = (public_pem, private_pem, key_password)
        return public_pem, private_pem
    
    def sign_data(self, peer_id: str, data: bytes) -> bytes:
        """Sign data with peer's private key"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("Cryptography libraries are required for secure signing")
            
        if peer_id not in self.key_pairs:
            raise ValueError(f"No key pair found for peer {peer_id}")
            
        # Extract encrypted private key and password
        _, private_key_pem, key_password = self.key_pairs[peer_id]
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=key_password
        )
        
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, peer_id: str, data: bytes, signature: bytes) -> bool:
        """Verify data signature"""
        if not CRYPTO_AVAILABLE:
            # Fallback verification
            try:
                signature_str = signature.decode()
                if signature_str.startswith(f"{peer_id}:"):
                    expected_hash = signature_str.split(":", 1)[1]
                    actual_hash = hashlib.sha256(data).hexdigest()
                    return expected_hash == actual_hash
            except:
                pass
            return False
            
        if peer_id not in self.key_pairs:
            return False
            
        # Real cryptographic verification
        public_key_pem, _ = self.key_pairs[peer_id]
        public_key = serialization.load_pem_public_key(public_key_pem)
        
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False
    
    def calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum"""
        return hashlib.sha256(data).hexdigest()
    
    def verify_checksum(self, data: bytes, checksum: str) -> bool:
        """Verify data integrity"""
        return self.calculate_checksum(data) == checksum


class ShaderCompressor:
    """Advanced shader compression with delta encoding"""
    
    def __init__(self):
        self.compression_stats = {
            'total_compressed': 0,
            'total_original_size': 0,
            'total_compressed_size': 0,
            'compression_times': deque(maxlen=100)
        }
        
        # Build compression dictionary from common shader patterns
        self.shader_dictionary = self._build_shader_dictionary()
    
    def _build_shader_dictionary(self) -> bytes:
        """Build compression dictionary for common shader patterns"""
        common_patterns = [
            b"gl_Position",
            b"gl_FragColor",
            b"gl_FragCoord",
            b"texture2D",
            b"uniform",
            b"varying",
            b"attribute",
            b"vec2",
            b"vec3",
            b"vec4",
            b"mat2",
            b"mat3", 
            b"mat4",
            b"float",
            b"int",
            b"bool",
            b"sampler2D",
            b"if",
            b"else",
            b"for",
            b"while",
            b"return",
            b"discard",
            b"normalize",
            b"dot",
            b"cross",
            b"length",
            b"distance",
            b"reflect",
            b"refract",
            b"mix",
            b"clamp",
            b"min",
            b"max",
            b"abs",
            b"sign",
            b"floor",
            b"ceil",
            b"fract",
            b"mod",
            b"pow",
            b"exp",
            b"log",
            b"sqrt",
            b"sin",
            b"cos",
            b"tan",
            b"asin",
            b"acos",
            b"atan"
        ]
        
        return b'\n'.join(common_patterns)
    
    def compress_shader(self, data: bytes, method: CompressionMethod = CompressionMethod.LZMA) -> Tuple[bytes, CompressionMethod]:
        """Compress shader data using specified method"""
        start_time = time.time()
        
        if method == CompressionMethod.NONE:
            compressed = data
            actual_method = CompressionMethod.NONE
        elif method == CompressionMethod.ZLIB:
            compressed = zlib.compress(data, level=6)  # Balanced compression
            actual_method = CompressionMethod.ZLIB
        elif method == CompressionMethod.LZMA:
            compressed = lzma.compress(
                data,
                format=lzma.FORMAT_ALONE,
                preset=1,  # Fast compression for Steam Deck
                check=lzma.CHECK_CRC32
            )
            actual_method = CompressionMethod.LZMA
        elif method == CompressionMethod.CUSTOM_DELTA:
            # Custom delta compression with dictionary
            compressed = self._delta_compress(data)
            actual_method = CompressionMethod.CUSTOM_DELTA
        else:
            # Fallback to zlib
            compressed = zlib.compress(data, level=6)
            actual_method = CompressionMethod.ZLIB
        
        # Choose best compression
        if len(compressed) >= len(data):
            # No benefit, use uncompressed
            compressed = data
            actual_method = CompressionMethod.NONE
        
        # Update statistics
        compression_time = time.time() - start_time
        self.compression_stats['total_compressed'] += 1
        self.compression_stats['total_original_size'] += len(data)
        self.compression_stats['total_compressed_size'] += len(compressed)
        self.compression_stats['compression_times'].append(compression_time)
        
        return compressed, actual_method
    
    def decompress_shader(self, data: bytes, method: CompressionMethod) -> bytes:
        """Decompress shader data"""
        if method == CompressionMethod.NONE:
            return data
        elif method == CompressionMethod.ZLIB:
            return zlib.decompress(data)
        elif method == CompressionMethod.LZMA:
            return lzma.decompress(data)
        elif method == CompressionMethod.CUSTOM_DELTA:
            return self._delta_decompress(data)
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def _delta_compress(self, data: bytes) -> bytes:
        """Custom delta compression with dictionary"""
        # Simple delta compression implementation
        # Replace common patterns with shorter tokens
        compressed = data
        
        # Replace common patterns with tokens
        token_map = {}
        token_counter = 0
        
        for pattern in self.shader_dictionary.split(b'\n'):
            if pattern and len(pattern) > 3:  # Only compress patterns > 3 bytes
                if pattern in data:
                    token = f"<T{token_counter}>".encode()
                    if len(token) < len(pattern):
                        token_map[pattern] = token
                        compressed = compressed.replace(pattern, token)
                        token_counter += 1
        
        # Prefix with token map for decompression
        token_map_data = pickle.dumps(token_map)
        token_map_size = len(token_map_data)
        
        result = struct.pack('<I', token_map_size) + token_map_data + compressed
        
        # Apply final compression
        return zlib.compress(result, level=3)
    
    def _delta_decompress(self, data: bytes) -> bytes:
        """Custom delta decompression"""
        # Decompress outer layer
        uncompressed = zlib.decompress(data)
        
        # Extract token map
        token_map_size = struct.unpack('<I', uncompressed[:4])[0]
        token_map_data = uncompressed[4:4 + token_map_size]
        compressed_data = uncompressed[4 + token_map_size:]
        
        token_map = pickle.loads(token_map_data)
        
        # Restore original patterns
        result = compressed_data
        for pattern, token in token_map.items():
            result = result.replace(token, pattern)
        
        return result
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics"""
        if self.compression_stats['total_original_size'] == 0:
            avg_ratio = 0.0
        else:
            avg_ratio = 1.0 - (
                self.compression_stats['total_compressed_size'] / 
                self.compression_stats['total_original_size']
            )
        
        avg_time = (
            sum(self.compression_stats['compression_times']) / 
            len(self.compression_stats['compression_times'])
            if self.compression_stats['compression_times'] else 0.0
        )
        
        return {
            'total_compressed': self.compression_stats['total_compressed'],
            'average_compression_ratio': avg_ratio,
            'average_compression_time_ms': avg_time * 1000,
            'total_size_saved_mb': (
                self.compression_stats['total_original_size'] - 
                self.compression_stats['total_compressed_size']
            ) / (1024 * 1024)
        }


class ReputationSystem:
    """Trust and reputation management for P2P network"""
    
    def __init__(self):
        self.peer_reputations = defaultdict(lambda: {
            'score': 0.0,
            'successful_transfers': 0,
            'failed_transfers': 0,
            'validation_successes': 0,
            'validation_failures': 0,
            'uptime_score': 0.0,
            'contribution_score': 0.0,
            'steam_deck_bonus': 0.0,
            'last_update': time.time()
        })
        
        self.reputation_weights = {
            'transfer_success_rate': 0.3,
            'validation_accuracy': 0.25,
            'uptime': 0.2,
            'contribution': 0.15,
            'steam_deck_bonus': 0.1
        }
        
        self.blacklisted_peers = set()
        
    def record_successful_transfer(self, peer_id: str, bytes_transferred: int):
        """Record successful data transfer"""
        rep = self.peer_reputations[peer_id]
        rep['successful_transfers'] += 1
        rep['contribution_score'] += bytes_transferred / (1024 * 1024)  # MB
        rep['last_update'] = time.time()
        self._update_reputation_score(peer_id)
    
    def record_failed_transfer(self, peer_id: str):
        """Record failed data transfer"""
        rep = self.peer_reputations[peer_id]
        rep['failed_transfers'] += 1
        rep['last_update'] = time.time()
        self._update_reputation_score(peer_id)
    
    def record_validation_result(self, peer_id: str, success: bool):
        """Record cache validation result"""
        rep = self.peer_reputations[peer_id]
        if success:
            rep['validation_successes'] += 1
        else:
            rep['validation_failures'] += 1
        rep['last_update'] = time.time()
        self._update_reputation_score(peer_id)
    
    def update_uptime_score(self, peer_id: str, hours_online: float):
        """Update peer uptime contribution"""
        rep = self.peer_reputations[peer_id]
        # Logarithmic uptime scoring
        rep['uptime_score'] = min(1.0, np.log(hours_online + 1) / np.log(168))  # Week = max
        self._update_reputation_score(peer_id)
    
    def set_steam_deck_verified(self, peer_id: str, verified: bool):
        """Set Steam Deck verification bonus"""
        rep = self.peer_reputations[peer_id]
        rep['steam_deck_bonus'] = 0.1 if verified else 0.0
        self._update_reputation_score(peer_id)
    
    def _update_reputation_score(self, peer_id: str):
        """Recalculate overall reputation score"""
        rep = self.peer_reputations[peer_id]
        
        # Calculate individual components
        total_transfers = rep['successful_transfers'] + rep['failed_transfers']
        transfer_success_rate = (
            rep['successful_transfers'] / max(1, total_transfers)
        )
        
        total_validations = rep['validation_successes'] + rep['validation_failures']
        validation_accuracy = (
            rep['validation_successes'] / max(1, total_validations)
        )
        
        # Normalize contribution score (logarithmic)
        contribution_norm = min(1.0, np.log(rep['contribution_score'] + 1) / np.log(1000))
        
        # Weighted combination
        score = (
            self.reputation_weights['transfer_success_rate'] * transfer_success_rate +
            self.reputation_weights['validation_accuracy'] * validation_accuracy +
            self.reputation_weights['uptime'] * rep['uptime_score'] +
            self.reputation_weights['contribution'] * contribution_norm +
            self.reputation_weights['steam_deck_bonus'] * rep['steam_deck_bonus']
        )
        
        # Apply penalties for bad behavior
        if rep['validation_failures'] > 10 or rep['failed_transfers'] > 50:
            score *= 0.5  # Severe penalty
        elif rep['validation_failures'] > 5 or rep['failed_transfers'] > 20:
            score *= 0.8  # Moderate penalty
        
        rep['score'] = max(0.0, min(1.0, score))
        
        # Blacklist consistently bad peers
        if rep['score'] < 0.1 and total_transfers > 10:
            self.blacklisted_peers.add(peer_id)
            logger.warning(f"Blacklisted peer {peer_id} due to poor reputation")
    
    def get_reputation_score(self, peer_id: str) -> float:
        """Get current reputation score for peer"""
        if peer_id in self.blacklisted_peers:
            return 0.0
        return self.peer_reputations[peer_id]['score']
    
    def is_trusted_peer(self, peer_id: str, min_score: float = 0.3) -> bool:
        """Check if peer is trusted"""
        return (
            peer_id not in self.blacklisted_peers and 
            self.get_reputation_score(peer_id) >= min_score
        )
    
    def get_top_peers(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top peers by reputation"""
        peer_scores = [
            (peer_id, rep['score'])
            for peer_id, rep in self.peer_reputations.items()
            if peer_id not in self.blacklisted_peers
        ]
        return sorted(peer_scores, key=lambda x: x[1], reverse=True)[:limit]
    
    def cleanup_old_entries(self, max_age_hours: float = 168):  # 1 week
        """Remove old reputation entries"""
        current_time = time.time()
        to_remove = []
        
        for peer_id, rep in self.peer_reputations.items():
            age_hours = (current_time - rep['last_update']) / 3600
            if age_hours > max_age_hours:
                to_remove.append(peer_id)
        
        for peer_id in to_remove:
            del self.peer_reputations[peer_id]
            self.blacklisted_peers.discard(peer_id)
        
        logger.info(f"Cleaned up {len(to_remove)} old reputation entries")


# Save the file and continue with the remaining components