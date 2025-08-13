"""
Distributed Hash Table (DHT) implementation for shader cache location
Based on Kademlia protocol optimized for Steam Deck P2P network
"""

import asyncio
import hashlib
import time
import random
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import struct
import socket
import logging
from enum import Enum
import bisect

# Import base P2P components
from p2p_shader_network import PeerInfo, P2PMessage, MessageType, ShaderCacheEntry

logger = logging.getLogger(__name__)


class DHTNodeState(Enum):
    """DHT node states"""
    JOINING = "joining"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    LEAVING = "leaving"


@dataclass
class DHTKey:
    """DHT key with utility methods"""
    key_bytes: bytes
    
    def __post_init__(self):
        if len(self.key_bytes) != 20:  # 160-bit keys
            raise ValueError("DHT keys must be 160 bits (20 bytes)")
    
    @classmethod
    def from_string(cls, s: str) -> 'DHTKey':
        """Create DHT key from string"""
        return cls(hashlib.sha1(s.encode()).digest())
    
    @classmethod
    def from_shader_hash(cls, shader_hash: str, game_id: str) -> 'DHTKey':
        """Create DHT key from shader hash and game ID"""
        combined = f"{game_id}:{shader_hash}"
        return cls.from_string(combined)
    
    def distance(self, other: 'DHTKey') -> int:
        """Calculate XOR distance to another key"""
        return int.from_bytes(
            bytes(a ^ b for a, b in zip(self.key_bytes, other.key_bytes)),
            'big'
        )
    
    def to_hex(self) -> str:
        """Convert to hex string"""
        return self.key_bytes.hex()
    
    def __lt__(self, other: 'DHTKey') -> bool:
        return self.key_bytes < other.key_bytes
    
    def __eq__(self, other: 'DHTKey') -> bool:
        return self.key_bytes == other.key_bytes
    
    def __hash__(self) -> int:
        return hash(self.key_bytes)
    
    def __str__(self) -> str:
        return self.to_hex()[:12] + "..."


@dataclass
class DHTNode:
    """DHT node information"""
    node_id: DHTKey
    peer_info: PeerInfo
    last_seen: float = field(default_factory=time.time)
    rtt_ms: float = 0.0  # Round trip time
    failure_count: int = 0
    
    def is_stale(self, max_age: float = 900) -> bool:  # 15 minutes
        """Check if node is stale"""
        return (time.time() - self.last_seen) > max_age
    
    def is_good(self) -> bool:
        """Check if node is in good state"""
        return self.failure_count < 3 and not self.is_stale()


@dataclass
class DHTValue:
    """Value stored in DHT"""
    key: DHTKey
    value: bytes
    timestamp: float
    ttl: float = 3600  # 1 hour default TTL
    publisher_id: str = ""
    signature: Optional[bytes] = None
    
    def is_expired(self) -> bool:
        """Check if value is expired"""
        return (time.time() - self.timestamp) > self.ttl
    
    def to_dict(self) -> Dict:
        return {
            'key': self.key.to_hex(),
            'value': self.value.hex(),
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'publisher_id': self.publisher_id
        }


class KBucket:
    """K-bucket for storing nodes at specific distance range"""
    
    def __init__(self, k: int = 20):
        self.k = k  # Bucket size (k-value)
        self.nodes: List[DHTNode] = []
        self.last_updated = time.time()
    
    def add_node(self, node: DHTNode) -> bool:
        """Add node to bucket"""
        # Update existing node
        for i, existing in enumerate(self.nodes):
            if existing.node_id == node.node_id:
                self.nodes[i] = node
                self.last_updated = time.time()
                return True
        
        # Add new node if bucket not full
        if len(self.nodes) < self.k:
            self.nodes.append(node)
            self.last_updated = time.time()
            return True
        
        # Replace bad node if exists
        for i, existing in enumerate(self.nodes):
            if not existing.is_good():
                self.nodes[i] = node
                self.last_updated = time.time()
                return True
        
        # Bucket is full with good nodes
        return False
    
    def remove_node(self, node_id: DHTKey) -> bool:
        """Remove node from bucket"""
        for i, node in enumerate(self.nodes):
            if node.node_id == node_id:
                del self.nodes[i]
                self.last_updated = time.time()
                return True
        return False
    
    def get_nodes(self, target: DHTKey, limit: int = None) -> List[DHTNode]:
        """Get nodes closest to target"""
        if limit is None:
            limit = self.k
            
        # Sort by distance to target
        nodes_with_distance = [
            (node, target.distance(node.node_id)) 
            for node in self.nodes 
            if node.is_good()
        ]
        
        nodes_with_distance.sort(key=lambda x: x[1])
        return [node for node, _ in nodes_with_distance[:limit]]
    
    def get_random_node(self) -> Optional[DHTNode]:
        """Get random node from bucket"""
        good_nodes = [node for node in self.nodes if node.is_good()]
        return random.choice(good_nodes) if good_nodes else None
    
    def cleanup_stale_nodes(self):
        """Remove stale nodes"""
        self.nodes = [node for node in self.nodes if not node.is_stale()]


class DHTRoutingTable:
    """Kademlia routing table"""
    
    def __init__(self, node_id: DHTKey, k: int = 20):
        self.node_id = node_id
        self.k = k
        self.buckets: List[KBucket] = [KBucket(k) for _ in range(160)]  # 160 bits
        self.last_maintenance = time.time()
    
    def _bucket_index(self, target: DHTKey) -> int:
        """Get bucket index for target key"""
        distance = self.node_id.distance(target)
        if distance == 0:
            return 0
        return 159 - distance.bit_length() + 1
    
    def add_node(self, node: DHTNode):
        """Add node to routing table"""
        if node.node_id == self.node_id:
            return  # Don't add ourselves
            
        bucket_index = self._bucket_index(node.node_id)
        bucket = self.buckets[bucket_index]
        
        added = bucket.add_node(node)
        if added:
            logger.debug(f"Added node {node.node_id} to bucket {bucket_index}")
    
    def remove_node(self, node_id: DHTKey):
        """Remove node from routing table"""
        bucket_index = self._bucket_index(node_id)
        bucket = self.buckets[bucket_index]
        
        removed = bucket.remove_node(node_id)
        if removed:
            logger.debug(f"Removed node {node_id} from bucket {bucket_index}")
    
    def find_closest_nodes(self, target: DHTKey, limit: int = 20) -> List[DHTNode]:
        """Find closest nodes to target"""
        all_nodes = []
        
        # Start with target's bucket and expand outward
        bucket_index = self._bucket_index(target)
        
        # Add nodes from target bucket
        all_nodes.extend(self.buckets[bucket_index].get_nodes(target))
        
        # Add nodes from nearby buckets if needed
        for offset in range(1, 160):
            if len(all_nodes) >= limit:
                break
                
            # Check lower bucket
            lower_index = bucket_index - offset
            if lower_index >= 0:
                all_nodes.extend(self.buckets[lower_index].get_nodes(target))
            
            # Check upper bucket
            upper_index = bucket_index + offset
            if upper_index < 160:
                all_nodes.extend(self.buckets[upper_index].get_nodes(target))
        
        # Sort by distance and return closest
        nodes_with_distance = [
            (node, target.distance(node.node_id)) 
            for node in all_nodes
        ]
        
        nodes_with_distance.sort(key=lambda x: x[1])
        return [node for node, _ in nodes_with_distance[:limit]]
    
    def get_random_nodes(self, count: int = 10) -> List[DHTNode]:
        """Get random nodes for maintenance"""
        nodes = []
        for bucket in self.buckets:
            node = bucket.get_random_node()
            if node:
                nodes.append(node)
        
        random.shuffle(nodes)
        return nodes[:count]
    
    def perform_maintenance(self):
        """Perform periodic maintenance"""
        current_time = time.time()
        
        # Skip if maintenance was recent
        if current_time - self.last_maintenance < 300:  # 5 minutes
            return
        
        logger.debug("Performing DHT maintenance")
        
        # Clean up stale nodes
        for bucket in self.buckets:
            bucket.cleanup_stale_nodes()
        
        self.last_maintenance = current_time
    
    def get_stats(self) -> Dict:
        """Get routing table statistics"""
        total_nodes = sum(len(bucket.nodes) for bucket in self.buckets)
        active_nodes = sum(
            len([node for node in bucket.nodes if node.is_good()]) 
            for bucket in self.buckets
        )
        
        bucket_counts = [len(bucket.nodes) for bucket in self.buckets]
        
        return {
            'total_nodes': total_nodes,
            'active_nodes': active_nodes,
            'max_bucket_size': max(bucket_counts) if bucket_counts else 0,
            'avg_bucket_size': sum(bucket_counts) / len(bucket_counts),
            'empty_buckets': bucket_counts.count(0),
            'last_maintenance': self.last_maintenance
        }


class ShaderDHT:
    """DHT implementation for shader cache location"""
    
    def __init__(self, node_id: str, port: int = 0):
        self.node_id = DHTKey.from_string(node_id)
        self.port = port
        self.state = DHTNodeState.JOINING
        
        # Routing table
        self.routing_table = DHTRoutingTable(self.node_id)
        
        # Local storage
        self.storage: Dict[DHTKey, DHTValue] = {}
        
        # Network transport
        self.transport = None
        self.pending_requests = {}  # request_id -> (future, timestamp)
        self.request_counter = 0
        
        # Statistics
        self.stats = {
            'lookups_performed': 0,
            'lookups_successful': 0,
            'stores_performed': 0,
            'stores_successful': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'bootstrap_attempts': 0,
            'network_errors': 0
        }
        
        # Configuration
        self.config = {
            'k': 20,  # Replication parameter
            'alpha': 3,  # Parallelism parameter
            'ttl_seconds': 3600,  # Default TTL
            'republish_interval': 3600,  # Republish every hour
            'maintenance_interval': 300,  # Maintenance every 5 minutes
            'request_timeout': 30,  # Request timeout in seconds
            'max_stored_values': 10000  # Maximum values to store
        }
        
        logger.info(f"DHT node initialized with ID: {self.node_id}")
    
    async def start(self, bootstrap_nodes: List[Tuple[str, int]] = None):
        """Start DHT node"""
        logger.info(f"Starting DHT node on port {self.port}")
        
        # Start network transport
        await self._start_transport()
        
        # Bootstrap if nodes provided
        if bootstrap_nodes:
            await self._bootstrap(bootstrap_nodes)
        
        # Start maintenance tasks
        asyncio.create_task(self._maintenance_loop())
        asyncio.create_task(self._republish_loop())
        
        self.state = DHTNodeState.ACTIVE
        logger.info("DHT node started successfully")
    
    async def stop(self):
        """Stop DHT node"""
        logger.info("Stopping DHT node")
        
        self.state = DHTNodeState.LEAVING
        
        if self.transport:
            self.transport.close()
        
        # Cancel pending requests
        for future, _ in self.pending_requests.values():
            future.cancel()
        
        self.pending_requests.clear()
        
        logger.info("DHT node stopped")
    
    async def store(self, key: DHTKey, value: bytes, ttl: float = None) -> bool:
        """Store value in DHT"""
        if ttl is None:
            ttl = self.config['ttl_seconds']
        
        self.stats['stores_performed'] += 1
        
        # Create DHT value
        dht_value = DHTValue(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl,
            publisher_id=self.node_id.to_hex()
        )
        
        # Find closest nodes
        closest_nodes = await self._lookup_nodes(key)
        
        if not closest_nodes:
            logger.warning(f"No nodes found for storing key {key}")
            return False
        
        # Store on closest k nodes
        store_tasks = []
        for node in closest_nodes[:self.config['k']]:
            task = self._send_store_request(node, dht_value)
            store_tasks.append(task)
        
        # Wait for responses
        try:
            results = await asyncio.gather(*store_tasks, return_exceptions=True)
            successful_stores = sum(1 for r in results if r is True)
            
            if successful_stores > 0:
                self.stats['stores_successful'] += 1
                logger.debug(f"Stored key {key} on {successful_stores} nodes")
                return True
            else:
                logger.warning(f"Failed to store key {key}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing key {key}: {e}")
            return False
    
    async def lookup(self, key: DHTKey) -> Optional[bytes]:
        """Lookup value in DHT"""
        self.stats['lookups_performed'] += 1
        
        # Check local storage first
        if key in self.storage:
            value = self.storage[key]
            if not value.is_expired():
                return value.value
            else:
                # Remove expired value
                del self.storage[key]
        
        # Perform network lookup
        closest_nodes = await self._lookup_nodes(key)
        
        if not closest_nodes:
            logger.debug(f"No nodes found for key {key}")
            return None
        
        # Query closest nodes for value
        for node in closest_nodes:
            try:
                value = await self._send_lookup_request(node, key)
                if value:
                    self.stats['lookups_successful'] += 1
                    logger.debug(f"Found value for key {key} from node {node.node_id}")
                    return value
            except Exception as e:
                logger.debug(f"Lookup failed for node {node.node_id}: {e}")
                continue
        
        logger.debug(f"Value not found for key {key}")
        return None
    
    async def lookup_shader_cache(self, shader_hash: str, game_id: str) -> Optional[ShaderCacheEntry]:
        """Lookup shader cache entry"""
        key = DHTKey.from_shader_hash(shader_hash, game_id)
        value_bytes = await self.lookup(key)
        
        if value_bytes:
            try:
                import pickle
                cache_entry = pickle.loads(value_bytes)
                return cache_entry
            except Exception as e:
                logger.error(f"Failed to deserialize shader cache: {e}")
        
        return None
    
    async def store_shader_cache(self, cache_entry: ShaderCacheEntry) -> bool:
        """Store shader cache entry in DHT"""
        key = DHTKey.from_shader_hash(cache_entry.shader_hash, cache_entry.game_id)
        
        try:
            import pickle
            value_bytes = pickle.dumps(cache_entry)
            return await self.store(key, value_bytes)
        except Exception as e:
            logger.error(f"Failed to serialize shader cache: {e}")
            return False
    
    async def _lookup_nodes(self, key: DHTKey) -> List[DHTNode]:
        """Perform iterative node lookup"""
        # Start with closest known nodes
        closest_nodes = self.routing_table.find_closest_nodes(key, self.config['k'])
        
        if not closest_nodes:
            return []
        
        # Track queried nodes and distances
        queried_nodes = set()
        unqueried_nodes = set(closest_nodes)
        
        while unqueried_nodes:
            # Select alpha closest unqueried nodes
            query_nodes = sorted(
                unqueried_nodes, 
                key=lambda n: key.distance(n.node_id)
            )[:self.config['alpha']]
            
            # Mark as queried
            for node in query_nodes:
                queried_nodes.add(node.node_id)
                unqueried_nodes.remove(node)
            
            # Query nodes in parallel
            query_tasks = [
                self._send_find_node_request(node, key) 
                for node in query_nodes
            ]
            
            try:
                results = await asyncio.gather(*query_tasks, return_exceptions=True)
                
                # Process responses
                for result in results:
                    if isinstance(result, list):
                        for new_node in result:
                            if (new_node.node_id not in queried_nodes and 
                                new_node.node_id != self.node_id):
                                
                                unqueried_nodes.add(new_node)
                                closest_nodes.append(new_node)
                
                # Keep only k closest nodes
                closest_nodes.sort(key=lambda n: key.distance(n.node_id))
                closest_nodes = closest_nodes[:self.config['k']]
                
                # Remove any unqueried nodes that are farther than our closest
                if closest_nodes:
                    max_distance = key.distance(closest_nodes[-1].node_id)
                    unqueried_nodes = {
                        node for node in unqueried_nodes
                        if key.distance(node.node_id) < max_distance
                    }
                    
            except Exception as e:
                logger.error(f"Error in node lookup: {e}")
                break
        
        return closest_nodes
    
    async def _send_find_node_request(self, node: DHTNode, key: DHTKey) -> List[DHTNode]:
        """Send FIND_NODE request"""
        try:
            message = P2PMessage(
                msg_type=MessageType.DHT_FIND,
                sender_id=self.node_id.to_hex(),
                recipient_id=node.peer_info.peer_id,
                payload={'key': key.to_hex(), 'type': 'node'}
            )
            
            response = await self._send_request(node, message)
            
            if response and 'nodes' in response.payload:
                nodes = []
                for node_data in response.payload['nodes']:
                    try:
                        peer_info = PeerInfo.from_dict(node_data)
                        dht_node = DHTNode(
                            node_id=DHTKey.from_string(peer_info.peer_id),
                            peer_info=peer_info
                        )
                        nodes.append(dht_node)
                    except Exception as e:
                        logger.debug(f"Invalid node data: {e}")
                
                return nodes
            
        except Exception as e:
            logger.debug(f"FIND_NODE request failed: {e}")
        
        return []
    
    async def _send_lookup_request(self, node: DHTNode, key: DHTKey) -> Optional[bytes]:
        """Send value lookup request"""
        try:
            message = P2PMessage(
                msg_type=MessageType.DHT_FIND,
                sender_id=self.node_id.to_hex(),
                recipient_id=node.peer_info.peer_id,
                payload={'key': key.to_hex(), 'type': 'value'}
            )
            
            response = await self._send_request(node, message)
            
            if response and 'value' in response.payload:
                value_hex = response.payload['value']
                return bytes.fromhex(value_hex)
                
        except Exception as e:
            logger.debug(f"Lookup request failed: {e}")
        
        return None
    
    async def _send_store_request(self, node: DHTNode, dht_value: DHTValue) -> bool:
        """Send store request"""
        try:
            message = P2PMessage(
                msg_type=MessageType.DHT_STORE,
                sender_id=self.node_id.to_hex(),
                recipient_id=node.peer_info.peer_id,
                payload=dht_value.to_dict()
            )
            
            response = await self._send_request(node, message)
            return response and response.payload.get('success', False)
            
        except Exception as e:
            logger.debug(f"Store request failed: {e}")
            return False
    
    async def _send_request(self, node: DHTNode, message: P2PMessage, timeout: float = None) -> Optional[P2PMessage]:
        """Send request and wait for response"""
        if timeout is None:
            timeout = self.config['request_timeout']
        
        # Generate request ID
        request_id = self.request_counter
        self.request_counter += 1
        message.sequence_id = request_id
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_requests[request_id] = (response_future, time.time())
        
        try:
            # Send message
            await self._send_message(node, message)
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.debug(f"Request {request_id} timed out")
            node.failure_count += 1
            return None
        except Exception as e:
            logger.debug(f"Request {request_id} failed: {e}")
            node.failure_count += 1
            return None
        finally:
            # Clean up request
            self.pending_requests.pop(request_id, None)
    
    async def _send_message(self, node: DHTNode, message: P2PMessage):
        """Send message to node"""
        # This would be implemented by the transport layer
        # For now, just log the message
        self.stats['messages_sent'] += 1
        logger.debug(f"Sending {message.msg_type} to {node.node_id}")
    
    def _handle_message(self, message: P2PMessage, sender_address: Tuple[str, int]):
        """Handle incoming DHT message"""
        self.stats['messages_received'] += 1
        
        # Add sender to routing table
        try:
            sender_peer_info = PeerInfo(
                peer_id=message.sender_id,
                address=sender_address[0],
                port=sender_address[1],
                role=PeerRole.LEECHER  # Default role
            )
            
            sender_node = DHTNode(
                node_id=DHTKey.from_string(message.sender_id),
                peer_info=sender_peer_info
            )
            
            self.routing_table.add_node(sender_node)
            
        except Exception as e:
            logger.debug(f"Error processing sender info: {e}")
        
        # Process message based on type
        if message.msg_type == MessageType.DHT_FIND:
            asyncio.create_task(self._handle_find_request(message))
        elif message.msg_type == MessageType.DHT_STORE:
            asyncio.create_task(self._handle_store_request(message))
        elif message.msg_type == MessageType.DHT_RESPONSE:
            self._handle_response(message)
        else:
            logger.debug(f"Unknown DHT message type: {message.msg_type}")
    
    async def _handle_find_request(self, message: P2PMessage):
        """Handle FIND_NODE or FIND_VALUE request"""
        try:
            key = DHTKey(bytes.fromhex(message.payload['key']))
            request_type = message.payload.get('type', 'node')
            
            response_payload = {}
            
            if request_type == 'value':
                # Check if we have the value
                if key in self.storage:
                    value = self.storage[key]
                    if not value.is_expired():
                        response_payload['value'] = value.value.hex()
                    else:
                        # Remove expired value
                        del self.storage[key]
            
            # Always include closest nodes
            if 'value' not in response_payload:
                closest_nodes = self.routing_table.find_closest_nodes(key, self.config['k'])
                response_payload['nodes'] = [node.peer_info.to_dict() for node in closest_nodes]
            
            # Send response
            response = P2PMessage(
                msg_type=MessageType.DHT_RESPONSE,
                sender_id=self.node_id.to_hex(),
                recipient_id=message.sender_id,
                payload=response_payload,
                sequence_id=message.sequence_id
            )
            
            # This would be sent via transport
            logger.debug(f"Sending DHT response to {message.sender_id}")
            
        except Exception as e:
            logger.error(f"Error handling find request: {e}")
    
    async def _handle_store_request(self, message: P2PMessage):
        """Handle STORE request"""
        try:
            key = DHTKey(bytes.fromhex(message.payload['key']))
            value = bytes.fromhex(message.payload['value'])
            ttl = message.payload.get('ttl', self.config['ttl_seconds'])
            
            # Create DHT value
            dht_value = DHTValue(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                publisher_id=message.payload.get('publisher_id', message.sender_id)
            )
            
            # Store locally if we have space
            success = False
            if len(self.storage) < self.config['max_stored_values']:
                self.storage[key] = dht_value
                success = True
                logger.debug(f"Stored value for key {key}")
            else:
                # Try to make space by removing expired values
                self._cleanup_expired_values()
                if len(self.storage) < self.config['max_stored_values']:
                    self.storage[key] = dht_value
                    success = True
                else:
                    logger.warning("Storage full, cannot store value")
            
            # Send response
            response = P2PMessage(
                msg_type=MessageType.DHT_RESPONSE,
                sender_id=self.node_id.to_hex(),
                recipient_id=message.sender_id,
                payload={'success': success},
                sequence_id=message.sequence_id
            )
            
            logger.debug(f"Sending store response to {message.sender_id}")
            
        except Exception as e:
            logger.error(f"Error handling store request: {e}")
    
    def _handle_response(self, message: P2PMessage):
        """Handle response message"""
        request_id = message.sequence_id
        
        if request_id in self.pending_requests:
            future, _ = self.pending_requests[request_id]
            if not future.done():
                future.set_result(message)
    
    def _cleanup_expired_values(self):
        """Remove expired values from storage"""
        expired_keys = [
            key for key, value in self.storage.items()
            if value.is_expired()
        ]
        
        for key in expired_keys:
            del self.storage[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired values")
    
    async def _bootstrap(self, bootstrap_nodes: List[Tuple[str, int]]):
        """Bootstrap DHT by joining network"""
        logger.info(f"Bootstrapping with {len(bootstrap_nodes)} nodes")
        
        self.stats['bootstrap_attempts'] += 1
        
        # Try to connect to bootstrap nodes
        connected_count = 0
        for address, port in bootstrap_nodes:
            try:
                # This would establish connection via transport
                logger.debug(f"Connecting to bootstrap node {address}:{port}")
                
                # Simulate adding bootstrap node to routing table
                bootstrap_peer = PeerInfo(
                    peer_id=f"bootstrap_{address}_{port}",
                    address=address,
                    port=port,
                    role=PeerRole.BOOTSTRAP
                )
                
                bootstrap_node = DHTNode(
                    node_id=DHTKey.from_string(bootstrap_peer.peer_id),
                    peer_info=bootstrap_peer
                )
                
                self.routing_table.add_node(bootstrap_node)
                connected_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap node {address}:{port}: {e}")
        
        if connected_count == 0:
            logger.error("Failed to connect to any bootstrap nodes")
            return
        
        # Perform lookup for our own ID to populate routing table
        await self._lookup_nodes(self.node_id)
        
        logger.info(f"Bootstrap complete, connected to {connected_count} nodes")
    
    async def _start_transport(self):
        """Start network transport"""
        # This would start the actual network transport
        # For now, just log
        logger.debug(f"Transport started on port {self.port}")
    
    async def _maintenance_loop(self):
        """Periodic maintenance loop"""
        while self.state == DHTNodeState.ACTIVE:
            try:
                await asyncio.sleep(self.config['maintenance_interval'])
                
                if self.state != DHTNodeState.ACTIVE:
                    break
                
                # Perform routing table maintenance
                self.routing_table.perform_maintenance()
                
                # Clean up expired values
                self._cleanup_expired_values()
                
                # Clean up old requests
                current_time = time.time()
                expired_requests = [
                    req_id for req_id, (_, timestamp) in self.pending_requests.items()
                    if current_time - timestamp > self.config['request_timeout']
                ]
                
                for req_id in expired_requests:
                    future, _ = self.pending_requests.pop(req_id)
                    if not future.done():
                        future.cancel()
                
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
    
    async def _republish_loop(self):
        """Republish stored values periodically"""
        while self.state == DHTNodeState.ACTIVE:
            try:
                await asyncio.sleep(self.config['republish_interval'])
                
                if self.state != DHTNodeState.ACTIVE:
                    break
                
                # Republish values that are about to expire
                current_time = time.time()
                
                for key, value in list(self.storage.items()):
                    time_until_expiry = value.ttl - (current_time - value.timestamp)
                    
                    # Republish if less than 1 hour until expiry
                    if 0 < time_until_expiry < 3600:
                        logger.debug(f"Republishing key {key}")
                        await self.store(key, value.value, value.ttl)
                
            except Exception as e:
                logger.error(f"Republish loop error: {e}")
    
    def get_stats(self) -> Dict:
        """Get DHT statistics"""
        routing_stats = self.routing_table.get_stats()
        
        return {
            'node_id': self.node_id.to_hex(),
            'state': self.state.value,
            'local_values': len(self.storage),
            'routing_table': routing_stats,
            'network': self.stats.copy(),
            'config': self.config.copy()
        }