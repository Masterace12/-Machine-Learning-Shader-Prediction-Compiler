"""
Bandwidth Optimization and Connection Management for Steam Deck P2P Network
Optimized for WiFi constraints and mobile networking conditions
"""

import asyncio
import time
import threading
from typing import Dict, List, Set, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
import socket
import struct
import random
import statistics

# Import P2P components
from p2p_shader_network import PeerInfo, PeerRole, MessageType, P2PMessage
from p2p_dht_system import DHTKey

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    THROTTLED = "throttled"
    FAILED = "failed"
    NAT_TRAVERSAL = "nat_traversal"


class BandwidthPriority(Enum):
    """Message priority levels for bandwidth allocation"""
    CRITICAL = 0    # DHT maintenance, heartbeats
    HIGH = 1        # Real-time shader requests
    NORMAL = 2      # Regular shader downloads
    LOW = 3         # Background sync, reputation updates
    BULK = 4        # Large cache transfers


@dataclass
class NetworkCondition:
    """Current network condition assessment"""
    bandwidth_kbps: float
    latency_ms: float
    packet_loss: float
    jitter_ms: float
    signal_strength: float  # 0-100 for WiFi
    connection_type: str    # "wifi", "mobile", "ethernet"
    is_metered: bool
    timestamp: float = field(default_factory=time.time)
    
    def get_quality_score(self) -> float:
        """Calculate overall network quality score (0-1)"""
        # Normalize metrics
        bandwidth_score = min(1.0, self.bandwidth_kbps / 10000)  # 10 Mbps = perfect
        latency_score = max(0.0, 1.0 - self.latency_ms / 500)   # 500ms = worst
        loss_score = max(0.0, 1.0 - self.packet_loss / 0.1)    # 10% = worst
        jitter_score = max(0.0, 1.0 - self.jitter_ms / 100)    # 100ms = worst
        signal_score = self.signal_strength / 100
        
        # Weighted average
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # bandwidth, latency, loss, jitter, signal
        scores = [bandwidth_score, latency_score, loss_score, jitter_score, signal_score]
        
        return sum(w * s for w, s in zip(weights, scores))


@dataclass
class TransferSession:
    """Active transfer session"""
    session_id: str
    peer_id: str
    direction: str  # "upload" or "download"
    total_bytes: int
    transferred_bytes: int = 0
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    priority: BandwidthPriority = BandwidthPriority.NORMAL
    chunk_size: int = 1024
    retry_count: int = 0
    
    def get_progress(self) -> float:
        """Get transfer progress (0-1)"""
        if self.total_bytes == 0:
            return 1.0
        return self.transferred_bytes / self.total_bytes
    
    def get_throughput_kbps(self) -> float:
        """Get current throughput in KB/s"""
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return (self.transferred_bytes / 1024) / elapsed
    
    def is_stalled(self, timeout: float = 30.0) -> bool:
        """Check if transfer is stalled"""
        return (time.time() - self.last_activity) > timeout


@dataclass
class PeerConnection:
    """Connection to a peer with bandwidth management"""
    peer_info: PeerInfo
    state: ConnectionState = ConnectionState.DISCONNECTED
    last_ping: float = 0.0
    rtt_ms: float = 0.0
    rtt_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Bandwidth tracking
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    
    # Rate limiting
    send_tokens: float = 0.0
    receive_tokens: float = 0.0
    last_token_update: float = field(default_factory=time.time)
    
    # Connection quality
    connection_score: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    
    # NAT traversal
    nat_type: str = "unknown"
    external_address: Optional[Tuple[str, int]] = None
    punch_attempts: int = 0
    
    def update_rtt(self, rtt_ms: float):
        """Update RTT measurement"""
        self.rtt_ms = rtt_ms
        self.rtt_history.append(rtt_ms)
        
        # Update connection score
        avg_rtt = statistics.mean(self.rtt_history)
        self.connection_score = max(0.0, 1.0 - (avg_rtt / 1000))  # 1s = worst
    
    def add_tokens(self, send_rate_kbps: float, receive_rate_kbps: float):
        """Add rate limiting tokens"""
        current_time = time.time()
        time_delta = current_time - self.last_token_update
        
        if time_delta > 0:
            # Add tokens based on allowed rates
            max_send_tokens = send_rate_kbps * 1024  # Convert to bytes
            max_receive_tokens = receive_rate_kbps * 1024
            
            self.send_tokens = min(
                max_send_tokens,
                self.send_tokens + (send_rate_kbps * 1024 * time_delta)
            )
            
            self.receive_tokens = min(
                max_receive_tokens,
                self.receive_tokens + (receive_rate_kbps * 1024 * time_delta)
            )
            
            self.last_token_update = current_time
    
    def can_send(self, bytes_count: int) -> bool:
        """Check if we can send bytes without violating rate limit"""
        return self.send_tokens >= bytes_count
    
    def consume_send_tokens(self, bytes_count: int):
        """Consume send tokens"""
        self.send_tokens = max(0, self.send_tokens - bytes_count)
        self.bytes_sent += bytes_count
    
    def consume_receive_tokens(self, bytes_count: int):
        """Consume receive tokens"""
        self.receive_tokens = max(0, self.receive_tokens - bytes_count)
        self.bytes_received += bytes_count


class BandwidthManager:
    """Manages bandwidth allocation and optimization"""
    
    def __init__(self, max_bandwidth_kbps: float = 2048):  # 2 Mbps default for Steam Deck WiFi
        self.max_bandwidth_kbps = max_bandwidth_kbps
        self.current_bandwidth_kbps = max_bandwidth_kbps
        self.network_condition = NetworkCondition(
            bandwidth_kbps=max_bandwidth_kbps,
            latency_ms=50,
            packet_loss=0.0,
            jitter_ms=5,
            signal_strength=80,
            connection_type="wifi",
            is_metered=False
        )
        
        # Bandwidth allocation by priority
        self.priority_allocation = {
            BandwidthPriority.CRITICAL: 0.15,  # 15% for critical
            BandwidthPriority.HIGH: 0.30,      # 30% for high priority
            BandwidthPriority.NORMAL: 0.35,    # 35% for normal
            BandwidthPriority.LOW: 0.15,       # 15% for low
            BandwidthPriority.BULK: 0.05       # 5% for bulk
        }
        
        # Current usage tracking
        self.usage_by_priority = defaultdict(float)  # KB/s
        self.active_transfers = {}  # session_id -> TransferSession
        
        # Adaptive parameters
        self.congestion_threshold = 0.8  # 80% utilization triggers congestion control
        self.throttle_factor = 0.7  # Reduce to 70% when throttling
        self.boost_factor = 1.2  # Boost to 120% when conditions are good
        
        # Network monitoring
        self.bandwidth_history = deque(maxlen=100)  # Last 100 measurements
        self.latency_history = deque(maxlen=50)
        self.last_measurement = time.time()
        
        logger.info(f"Bandwidth manager initialized with {max_bandwidth_kbps} KB/s limit")
    
    def update_network_condition(self, condition: NetworkCondition):
        """Update current network condition"""
        self.network_condition = condition
        self.bandwidth_history.append(condition.bandwidth_kbps)
        self.latency_history.append(condition.latency_ms)
        
        # Adjust current bandwidth based on conditions
        quality_score = condition.get_quality_score()
        
        if quality_score > 0.8:
            # Good conditions, can boost
            self.current_bandwidth_kbps = min(
                self.max_bandwidth_kbps * self.boost_factor,
                condition.bandwidth_kbps
            )
        elif quality_score < 0.3:
            # Poor conditions, throttle aggressively  
            self.current_bandwidth_kbps = condition.bandwidth_kbps * self.throttle_factor
        else:
            # Normal conditions
            self.current_bandwidth_kbps = condition.bandwidth_kbps
        
        logger.debug(f"Network condition updated: quality={quality_score:.2f}, "
                    f"bandwidth={self.current_bandwidth_kbps:.0f} KB/s")
    
    def allocate_bandwidth(self, priority: BandwidthPriority) -> float:
        """Get allocated bandwidth for priority level in KB/s"""
        allocation_ratio = self.priority_allocation[priority]
        base_allocation = self.current_bandwidth_kbps * allocation_ratio
        
        # Apply dynamic adjustments based on current usage
        total_usage = sum(self.usage_by_priority.values())
        available_bandwidth = max(0, self.current_bandwidth_kbps - total_usage)
        
        # If we're under-utilizing, allow higher priorities to use more
        if total_usage < self.current_bandwidth_kbps * 0.5:
            if priority in [BandwidthPriority.CRITICAL, BandwidthPriority.HIGH]:
                base_allocation += available_bandwidth * 0.3
        
        return base_allocation
    
    def request_bandwidth(self, session_id: str, bytes_per_second: float, 
                         priority: BandwidthPriority) -> bool:
        """Request bandwidth allocation for transfer"""
        allocated_bandwidth = self.allocate_bandwidth(priority)
        current_usage = self.usage_by_priority[priority]
        
        # Check if request can be satisfied
        if current_usage + bytes_per_second <= allocated_bandwidth:
            self.usage_by_priority[priority] += bytes_per_second
            logger.debug(f"Allocated {bytes_per_second:.0f} KB/s for {session_id} "
                        f"(priority: {priority.name})")
            return True
        
        logger.debug(f"Bandwidth request denied for {session_id}: "
                    f"requested={bytes_per_second:.0f}, "
                    f"available={allocated_bandwidth - current_usage:.0f}")
        return False
    
    def release_bandwidth(self, session_id: str, bytes_per_second: float,
                         priority: BandwidthPriority):
        """Release bandwidth allocation"""
        self.usage_by_priority[priority] = max(
            0, self.usage_by_priority[priority] - bytes_per_second
        )
        logger.debug(f"Released {bytes_per_second:.0f} KB/s for {session_id}")
    
    def get_optimal_chunk_size(self, priority: BandwidthPriority, rtt_ms: float) -> int:
        """Calculate optimal chunk size based on network conditions"""
        base_chunk = 1024  # 1KB base
        
        # Adjust for RTT (larger chunks for higher latency)
        rtt_factor = min(8.0, max(0.5, rtt_ms / 50))  # 50ms = 1x factor
        
        # Adjust for priority
        priority_factors = {
            BandwidthPriority.CRITICAL: 0.5,  # Smaller chunks for reliability
            BandwidthPriority.HIGH: 0.8,
            BandwidthPriority.NORMAL: 1.0,
            BandwidthPriority.LOW: 1.5,
            BandwidthPriority.BULK: 2.0      # Larger chunks for efficiency
        }
        
        priority_factor = priority_factors.get(priority, 1.0)
        
        # Adjust for bandwidth (larger chunks for higher bandwidth)
        bandwidth_factor = min(4.0, max(0.25, self.current_bandwidth_kbps / 1024))
        
        # Adjust for packet loss (smaller chunks for lossy connections)
        loss_factor = max(0.2, 1.0 - self.network_condition.packet_loss * 5)
        
        optimal_size = int(base_chunk * rtt_factor * priority_factor * 
                          bandwidth_factor * loss_factor)
        
        # Clamp to reasonable bounds
        return max(512, min(65536, optimal_size))  # 512B to 64KB
    
    def should_compress(self, data_size: int, priority: BandwidthPriority) -> bool:
        """Decide if data should be compressed based on conditions"""
        # Always compress for low bandwidth or high priority
        if (self.current_bandwidth_kbps < 512 or 
            priority in [BandwidthPriority.CRITICAL, BandwidthPriority.HIGH]):
            return data_size > 256  # Compress anything over 256 bytes
        
        # For normal/low priority, only compress larger data
        if priority == BandwidthPriority.BULK:
            return data_size > 1024  # 1KB threshold for bulk
        
        return data_size > 512  # 512B threshold for others
    
    def get_retry_delay(self, attempt: int, priority: BandwidthPriority) -> float:
        """Calculate retry delay with exponential backoff"""
        base_delays = {
            BandwidthPriority.CRITICAL: 0.5,
            BandwidthPriority.HIGH: 1.0,
            BandwidthPriority.NORMAL: 2.0,
            BandwidthPriority.LOW: 5.0,
            BandwidthPriority.BULK: 10.0
        }
        
        base_delay = base_delays.get(priority, 2.0)
        
        # Exponential backoff with jitter
        delay = base_delay * (2 ** min(attempt, 6))  # Cap at 2^6
        jitter = random.uniform(0.8, 1.2)  # ±20% jitter
        
        return delay * jitter
    
    def is_congested(self) -> bool:
        """Check if network is congested"""
        total_usage = sum(self.usage_by_priority.values())
        utilization = total_usage / max(1, self.current_bandwidth_kbps)
        return utilization > self.congestion_threshold
    
    def get_stats(self) -> Dict:
        """Get bandwidth manager statistics"""
        total_usage = sum(self.usage_by_priority.values())
        utilization = total_usage / max(1, self.current_bandwidth_kbps)
        
        return {
            'max_bandwidth_kbps': self.max_bandwidth_kbps,
            'current_bandwidth_kbps': self.current_bandwidth_kbps,
            'total_usage_kbps': total_usage,
            'utilization': utilization,
            'is_congested': self.is_congested(),
            'network_condition': {
                'quality_score': self.network_condition.get_quality_score(),
                'bandwidth_kbps': self.network_condition.bandwidth_kbps,
                'latency_ms': self.network_condition.latency_ms,
                'packet_loss': self.network_condition.packet_loss,
                'connection_type': self.network_condition.connection_type,
                'is_metered': self.network_condition.is_metered
            },
            'usage_by_priority': dict(self.usage_by_priority),
            'active_transfers': len(self.active_transfers)
        }


class ConnectionManager:
    """Manages P2P connections with optimization for Steam Deck"""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.connections: Dict[str, PeerConnection] = {}
        self.bandwidth_manager = BandwidthManager()
        
        # Connection prioritization
        self.role_priorities = {
            PeerRole.BOOTSTRAP: 10,
            PeerRole.SEEDER: 8,
            PeerRole.RELAY: 6,
            PeerRole.LEECHER: 4
        }
        
        # Connection maintenance
        self.maintenance_interval = 30.0  # seconds
        self.ping_interval = 60.0  # seconds
        self.connection_timeout = 300.0  # 5 minutes
        
        # NAT traversal
        self.nat_type = "unknown"
        self.external_address = None
        self.stun_servers = [
            ("stun.l.google.com", 19302),
            ("stun1.l.google.com", 19302),
            ("stun2.l.google.com", 19302)
        ]
        
        # Statistics
        self.stats = {
            'connections_attempted': 0,
            'connections_successful': 0,
            'connections_failed': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'nat_traversal_attempts': 0,
            'nat_traversal_successful': 0
        }
        
        logger.info(f"Connection manager initialized (max connections: {max_connections})")
    
    async def start(self):
        """Start connection manager"""
        logger.info("Starting connection manager")
        
        # Start maintenance tasks
        asyncio.create_task(self._maintenance_loop())
        asyncio.create_task(self._ping_loop())
        asyncio.create_task(self._network_monitor_loop())
        
        # Detect NAT type
        await self._detect_nat_type()
    
    async def connect_to_peer(self, peer_info: PeerInfo, priority: int = 0) -> bool:
        """Connect to a peer with bandwidth optimization"""
        peer_id = peer_info.peer_id
        
        if peer_id in self.connections:
            connection = self.connections[peer_id]
            if connection.state == ConnectionState.CONNECTED:
                return True
        
        self.stats['connections_attempted'] += 1
        
        # Check connection limits
        if len(self.connections) >= self.max_connections:
            # Try to evict lowest priority connection
            if not self._evict_connection():
                logger.warning("Cannot connect: connection limit reached")
                return False
        
        # Create new connection
        connection = PeerConnection(peer_info=peer_info)
        self.connections[peer_id] = connection
        
        # Attempt connection
        try:
            connection.state = ConnectionState.CONNECTING
            
            # Try direct connection first
            if await self._attempt_direct_connection(connection):
                connection.state = ConnectionState.CONNECTED
                self.stats['connections_successful'] += 1
                
                # Set up rate limiting based on peer role and network conditions
                self._setup_rate_limiting(connection)
                
                logger.info(f"Connected to peer {peer_id} ({peer_info.address}:{peer_info.port})")
                return True
            
            # Try NAT traversal if direct connection failed
            if await self._attempt_nat_traversal(connection):
                connection.state = ConnectionState.CONNECTED
                self.stats['connections_successful'] += 1
                self._setup_rate_limiting(connection)
                
                logger.info(f"Connected to peer {peer_id} via NAT traversal")
                return True
            
            # Connection failed
            connection.state = ConnectionState.FAILED
            connection.failure_count += 1
            self.stats['connections_failed'] += 1
            
            logger.warning(f"Failed to connect to peer {peer_id}")
            return False
            
        except Exception as e:
            connection.state = ConnectionState.FAILED
            self.stats['connections_failed'] += 1
            logger.error(f"Connection error for peer {peer_id}: {e}")
            return False
    
    async def disconnect_peer(self, peer_id: str):
        """Disconnect from a peer"""
        if peer_id in self.connections:
            connection = self.connections[peer_id]
            connection.state = ConnectionState.DISCONNECTED
            
            # Clean up any active transfers
            self._cleanup_peer_transfers(peer_id)
            
            del self.connections[peer_id]
            logger.info(f"Disconnected from peer {peer_id}")
    
    async def send_message(self, peer_id: str, message: P2PMessage, 
                          priority: BandwidthPriority = BandwidthPriority.NORMAL) -> bool:
        """Send message to peer with bandwidth management"""
        if peer_id not in self.connections:
            logger.warning(f"Cannot send message: not connected to {peer_id}")
            return False
        
        connection = self.connections[peer_id]
        
        if connection.state != ConnectionState.CONNECTED:
            logger.warning(f"Cannot send message: connection to {peer_id} not active")
            return False
        
        # Serialize message
        message_bytes = message.to_bytes()
        message_size = len(message_bytes)
        
        # Check rate limiting
        if not connection.can_send(message_size):
            logger.debug(f"Rate limited: cannot send to {peer_id}")
            return False
        
        try:
            # Apply compression if beneficial
            if self.bandwidth_manager.should_compress(message_size, priority):
                import zlib
                compressed_bytes = zlib.compress(message_bytes)
                if len(compressed_bytes) < message_size:
                    message_bytes = compressed_bytes
                    message_size = len(compressed_bytes)
            
            # Send message (this would be implemented by transport layer)
            await self._transport_send(connection, message_bytes)
            
            # Update statistics and rate limiting
            connection.consume_send_tokens(message_size)
            connection.messages_sent += 1
            self.stats['messages_sent'] += 1
            self.stats['bytes_sent'] += message_size
            
            logger.debug(f"Sent {message.msg_type} to {peer_id} ({message_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
            connection.failure_count += 1
            return False
    
    async def send_data_chunked(self, peer_id: str, data: bytes, 
                               priority: BandwidthPriority = BandwidthPriority.NORMAL) -> bool:
        """Send large data using chunked transfer"""
        if peer_id not in self.connections:
            return False
        
        connection = self.connections[peer_id]
        
        # Calculate optimal chunk size
        chunk_size = self.bandwidth_manager.get_optimal_chunk_size(
            priority, connection.rtt_ms
        )
        
        # Create transfer session
        session_id = f"{peer_id}_{int(time.time() * 1000)}"
        session = TransferSession(
            session_id=session_id,
            peer_id=peer_id,
            direction="upload",
            total_bytes=len(data),
            priority=priority,
            chunk_size=chunk_size
        )
        
        self.bandwidth_manager.active_transfers[session_id] = session
        
        try:
            # Send data in chunks
            offset = 0
            while offset < len(data):
                chunk_data = data[offset:offset + chunk_size]
                
                # Create chunk message
                chunk_message = P2PMessage(
                    msg_type=MessageType.CACHE_RESPONSE,  # Or appropriate type
                    sender_id="local",
                    recipient_id=peer_id,
                    payload={
                        'session_id': session_id,
                        'offset': offset,
                        'data': chunk_data.hex(),
                        'total_size': len(data),
                        'chunk_index': offset // chunk_size
                    }
                )
                
                # Send chunk with retry logic
                success = await self._send_chunk_with_retry(
                    connection, chunk_message, session
                )
                
                if not success:
                    logger.error(f"Failed to send chunk {offset} to {peer_id}")
                    return False
                
                offset += len(chunk_data)
                session.transferred_bytes = offset
                session.last_activity = time.time()
                
                # Adaptive delay to maintain target rate
                await self._adaptive_delay(session)
            
            logger.info(f"Completed chunked transfer to {peer_id} "
                       f"({len(data)} bytes in {session.get_throughput_kbps():.1f} KB/s)")
            return True
            
        except Exception as e:
            logger.error(f"Chunked transfer failed: {e}")
            return False
        finally:
            # Clean up session
            self.bandwidth_manager.active_transfers.pop(session_id, None)
    
    def get_best_peers(self, count: int = 5, role: PeerRole = None) -> List[PeerInfo]:
        """Get best connected peers based on connection quality"""
        eligible_connections = []
        
        for connection in self.connections.values():
            if connection.state != ConnectionState.CONNECTED:
                continue
            
            if role and connection.peer_info.role != role:
                continue
            
            # Calculate composite score
            role_priority = self.role_priorities.get(connection.peer_info.role, 0)
            reputation = connection.peer_info.reputation_score
            connection_quality = connection.connection_score
            
            composite_score = (
                role_priority * 0.3 +
                reputation * 0.4 +
                connection_quality * 0.3
            )
            
            eligible_connections.append((connection.peer_info, composite_score))
        
        # Sort by score and return top peers
        eligible_connections.sort(key=lambda x: x[1], reverse=True)
        return [peer_info for peer_info, _ in eligible_connections[:count]]
    
    async def _attempt_direct_connection(self, connection: PeerConnection) -> bool:
        """Attempt direct connection to peer"""
        try:
            peer_info = connection.peer_info
            
            # Simulate connection attempt
            start_time = time.time()
            
            # This would establish actual network connection
            await asyncio.sleep(0.1)  # Simulate connection delay
            
            # Measure RTT
            rtt_ms = (time.time() - start_time) * 1000
            connection.update_rtt(rtt_ms)
            
            return True  # Simulate successful connection
            
        except Exception as e:
            logger.debug(f"Direct connection failed: {e}")
            return False
    
    async def _attempt_nat_traversal(self, connection: PeerConnection) -> bool:
        """Attempt NAT traversal (hole punching)"""
        self.stats['nat_traversal_attempts'] += 1
        connection.punch_attempts += 1
        
        try:
            # Simulate NAT traversal process
            logger.debug(f"Attempting NAT traversal for {connection.peer_info.peer_id}")
            
            # This would implement actual hole punching
            await asyncio.sleep(0.2)  # Simulate traversal delay
            
            # Simulate success rate based on NAT types
            success_rate = 0.7  # 70% success rate
            if random.random() < success_rate:
                self.stats['nat_traversal_successful'] += 1
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"NAT traversal failed: {e}")
            return False
    
    def _setup_rate_limiting(self, connection: PeerConnection):
        """Set up rate limiting for connection"""
        peer_role = connection.peer_info.role
        
        # Base rates by role (KB/s)
        role_rates = {
            PeerRole.BOOTSTRAP: (100, 100),    # (send, receive)
            PeerRole.SEEDER: (512, 1024),      # High capacity
            PeerRole.RELAY: (256, 256),        # Balanced
            PeerRole.LEECHER: (128, 512)       # Typical Steam Deck
        }
        
        send_rate, receive_rate = role_rates.get(peer_role, (128, 512))
        
        # Adjust for network conditions
        quality_score = self.bandwidth_manager.network_condition.get_quality_score()
        send_rate *= quality_score
        receive_rate *= quality_score
        
        # Initialize token buckets
        connection.send_tokens = send_rate * 1024  # bytes
        connection.receive_tokens = receive_rate * 1024
        
        logger.debug(f"Rate limiting setup for {connection.peer_info.peer_id}: "
                    f"{send_rate:.0f}/{receive_rate:.0f} KB/s")
    
    def _evict_connection(self) -> bool:
        """Evict lowest priority connection"""
        if not self.connections:
            return False
        
        # Find connection with lowest score
        worst_connection = None
        worst_score = float('inf')
        
        for connection in self.connections.values():
            if connection.state != ConnectionState.CONNECTED:
                continue
            
            # Score based on role, reputation, and connection quality
            role_priority = self.role_priorities.get(connection.peer_info.role, 0)
            reputation = connection.peer_info.reputation_score
            connection_quality = connection.connection_score
            
            score = role_priority + reputation + connection_quality
            
            if score < worst_score:
                worst_score = score
                worst_connection = connection
        
        if worst_connection:
            asyncio.create_task(self.disconnect_peer(worst_connection.peer_info.peer_id))
            return True
        
        return False
    
    async def _send_chunk_with_retry(self, connection: PeerConnection, 
                                   message: P2PMessage, session: TransferSession) -> bool:
        """Send chunk with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                success = await self.send_message(
                    connection.peer_info.peer_id, message, session.priority
                )
                
                if success:
                    return True
                
                # Wait before retry
                if attempt < max_retries - 1:
                    delay = self.bandwidth_manager.get_retry_delay(attempt, session.priority)
                    await asyncio.sleep(delay)
                
            except Exception as e:
                logger.debug(f"Chunk send attempt {attempt + 1} failed: {e}")
        
        session.retry_count += 1
        return False
    
    async def _adaptive_delay(self, session: TransferSession):
        """Apply adaptive delay to maintain target transfer rate"""
        current_rate = session.get_throughput_kbps()
        
        # Target rate based on priority and network conditions
        target_rates = {
            BandwidthPriority.CRITICAL: 100,   # KB/s
            BandwidthPriority.HIGH: 200,
            BandwidthPriority.NORMAL: 150,
            BandwidthPriority.LOW: 100,
            BandwidthPriority.BULK: 50
        }
        
        target_rate = target_rates.get(session.priority, 100)
        
        # Adjust for network conditions
        network_factor = self.bandwidth_manager.network_condition.get_quality_score()
        target_rate *= network_factor
        
        if current_rate > target_rate * 1.2:  # 20% tolerance
            # Slow down
            delay = (current_rate - target_rate) / target_rate * 0.01
            await asyncio.sleep(min(0.1, delay))  # Max 100ms delay
    
    async def _transport_send(self, connection: PeerConnection, data: bytes):
        """Transport layer send (to be implemented)"""
        # This would be implemented by actual transport mechanism
        await asyncio.sleep(0.001)  # Simulate send delay
    
    def _cleanup_peer_transfers(self, peer_id: str):
        """Clean up transfers for disconnected peer"""
        to_remove = []
        for session_id, session in self.bandwidth_manager.active_transfers.items():
            if session.peer_id == peer_id:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.bandwidth_manager.active_transfers[session_id]
    
    async def _detect_nat_type(self):
        """Detect NAT type using STUN"""
        logger.info("Detecting NAT type...")
        
        try:
            # This would implement actual STUN protocol
            # For now, simulate detection
            await asyncio.sleep(0.5)
            
            nat_types = ["cone", "symmetric", "restricted", "port_restricted"]
            self.nat_type = random.choice(nat_types)
            self.external_address = ("203.0.113.1", random.randint(10000, 65535))
            
            logger.info(f"Detected NAT type: {self.nat_type}, "
                       f"external address: {self.external_address}")
            
        except Exception as e:
            logger.warning(f"NAT detection failed: {e}")
            self.nat_type = "unknown"
    
    async def _maintenance_loop(self):
        """Connection maintenance loop"""
        while True:
            try:
                await asyncio.sleep(self.maintenance_interval)
                
                current_time = time.time()
                stale_connections = []
                
                # Check for stale connections
                for peer_id, connection in self.connections.items():
                    if connection.state == ConnectionState.CONNECTED:
                        # Update rate limiting tokens
                        connection.add_tokens(128, 512)  # Default rates
                        
                        # Check for timeout
                        if (current_time - connection.last_ping) > self.connection_timeout:
                            stale_connections.append(peer_id)
                    
                    elif connection.state == ConnectionState.FAILED:
                        if connection.failure_count > 5:
                            stale_connections.append(peer_id)
                
                # Remove stale connections
                for peer_id in stale_connections:
                    await self.disconnect_peer(peer_id)
                
                logger.debug(f"Maintenance: {len(self.connections)} active connections, "
                            f"removed {len(stale_connections)} stale")
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
    
    async def _ping_loop(self):
        """Ping connected peers to measure RTT"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                ping_tasks = []
                for connection in self.connections.values():
                    if connection.state == ConnectionState.CONNECTED:
                        ping_tasks.append(self._ping_peer(connection))
                
                if ping_tasks:
                    await asyncio.gather(*ping_tasks, return_exceptions=True)
                
            except Exception as e:
                logger.error(f"Ping loop error: {e}")
    
    async def _ping_peer(self, connection: PeerConnection):
        """Ping individual peer"""
        try:
            start_time = time.time()
            
            ping_message = P2PMessage(
                msg_type=MessageType.PING,
                sender_id="local",
                recipient_id=connection.peer_info.peer_id,
                payload={'timestamp': start_time}
            )
            
            # Send ping (this would wait for pong response)
            await self.send_message(connection.peer_info.peer_id, ping_message, 
                                  BandwidthPriority.CRITICAL)
            
            # Simulate pong response
            await asyncio.sleep(0.05)  # Simulate network delay
            
            rtt_ms = (time.time() - start_time) * 1000
            connection.update_rtt(rtt_ms)
            connection.last_ping = time.time()
            connection.success_count += 1
            
        except Exception as e:
            logger.debug(f"Ping failed for {connection.peer_info.peer_id}: {e}")
            connection.failure_count += 1
    
    async def _network_monitor_loop(self):
        """Monitor network conditions"""
        while True:
            try:
                await asyncio.sleep(10.0)  # Monitor every 10 seconds
                
                # Simulate network monitoring
                # In real implementation, this would measure actual network metrics
                
                condition = NetworkCondition(
                    bandwidth_kbps=random.uniform(512, 2048),  # Vary bandwidth
                    latency_ms=random.uniform(20, 100),        # Vary latency
                    packet_loss=random.uniform(0, 0.05),       # 0-5% loss
                    jitter_ms=random.uniform(1, 20),           # Vary jitter
                    signal_strength=random.uniform(60, 100),   # WiFi signal
                    connection_type="wifi",
                    is_metered=False
                )
                
                self.bandwidth_manager.update_network_condition(condition)
                
            except Exception as e:
                logger.error(f"Network monitor error: {e}")
    
    def get_stats(self) -> Dict:
        """Get connection manager statistics"""
        connected_count = sum(
            1 for conn in self.connections.values() 
            if conn.state == ConnectionState.CONNECTED
        )
        
        return {
            'total_connections': len(self.connections),
            'connected': connected_count,
            'max_connections': self.max_connections,
            'nat_type': self.nat_type,
            'external_address': self.external_address,
            'network_stats': self.stats.copy(),
            'bandwidth_manager': self.bandwidth_manager.get_stats()
        }