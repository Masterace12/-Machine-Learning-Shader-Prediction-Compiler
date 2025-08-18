"""
Complete P2P Shader Cache Distribution System for Steam Deck
Integrates all components for secure, efficient shader sharing
"""

import asyncio
import time
import json
import threading
import ssl
from typing import Dict, List, Set, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from pathlib import Path
import logging
import random
import hashlib
import pickle

# Import all P2P components
from p2p_shader_network import (
    PeerInfo, PeerRole, ShaderCacheEntry, CompressionMethod,
    CryptographicValidator, ShaderCompressor, ReputationSystem
)
from p2p_dht_system import ShaderDHT, DHTKey
from p2p_bandwidth_optimizer import ConnectionManager, BandwidthPriority, NetworkCondition

# Import existing ML system
from shader_prediction_system import (
    SteamDeckShaderPredictor, ShaderMetrics, ShaderType,
    GameplayPatternAnalyzer
)
from steam_deck_integration import SteamDeckOptimizedSystem

logger = logging.getLogger(__name__)


@dataclass
class P2PShaderRequest:
    """Request for shader from P2P network"""
    shader_hash: str
    game_id: str
    shader_type: ShaderType
    priority: BandwidthPriority
    requestor_id: str
    timestamp: float = field(default_factory=time.time)
    max_wait_time: float = 30.0  # Maximum time to wait for response
    
    def get_cache_key(self) -> str:
        return f"{self.game_id}:{self.shader_hash}"


@dataclass
class P2PConfig:
    """Configuration for P2P shader distribution"""
    # Network settings
    listen_port: int = 0  # 0 = auto-select
    max_connections: int = 50
    max_bandwidth_kbps: float = 2048  # 2 Mbps for Steam Deck WiFi
    
    # DHT settings
    dht_k: int = 20
    dht_ttl_hours: float = 24
    
    # Cache settings
    max_cache_size_mb: float = 1024  # 1 GB cache limit
    cache_cleanup_interval: int = 3600  # 1 hour
    compression_enabled: bool = True
    
    # Security settings
    signature_verification: bool = True
    reputation_threshold: float = 0.3
    
    # Bootstrap nodes
    bootstrap_nodes: List[Tuple[str, int]] = field(default_factory=list)
    
    # Community features
    community_learning: bool = True
    contribution_rewards: bool = True


class P2PShaderDistributionSystem:
    """Main P2P shader distribution system"""
    
    def __init__(self, peer_id: str, config: P2PConfig = None):
        self.peer_id = peer_id
        self.config = config or P2PConfig()
        
        # Core components
        self.dht = ShaderDHT(peer_id, self.config.listen_port)
        self.connection_manager = ConnectionManager(self.config.max_connections)
        self.crypto_validator = CryptographicValidator()
        self.shader_compressor = ShaderCompressor()
        self.reputation_system = ReputationSystem()
        
        # ML integration
        self.ml_predictor: Optional[SteamDeckShaderPredictor] = None
        self.pattern_analyzer = GameplayPatternAnalyzer()
        
        # Local cache
        self.local_cache: Dict[str, ShaderCacheEntry] = {}
        self.cache_access_times = defaultdict(float)
        self.cache_size_bytes = 0
        
        # Request tracking
        self.pending_requests: Dict[str, P2PShaderRequest] = {}
        self.request_callbacks: Dict[str, Callable] = {}
        
        # Community learning data
        self.community_data = {
            'shader_usage_patterns': defaultdict(list),
            'performance_metrics': defaultdict(list),
            'optimization_suggestions': defaultdict(list)
        }
        
        # Statistics
        self.stats = {
            'shaders_requested': 0,
            'shaders_found_locally': 0,
            'shaders_found_p2p': 0,
            'shaders_shared': 0,
            'bytes_downloaded': 0,
            'bytes_uploaded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'community_contributions': 0,
            'start_time': time.time()
        }
        
        # State
        self.is_running = False
        self.peer_info = PeerInfo(
            peer_id=peer_id,
            address="0.0.0.0",  # Will be updated
            port=self.config.listen_port,
            role=PeerRole.LEECHER,  # Default role
            steam_deck_verified=True
        )
        
        logger.info(f"P2P shader distribution system initialized for peer {peer_id}")
    
    async def start(self, ml_predictor: SteamDeckShaderPredictor = None):
        """Start P2P system"""
        logger.info("Starting P2P shader distribution system")
        
        self.is_running = True
        self.ml_predictor = ml_predictor
        
        # Generate cryptographic keys
        public_key, private_key = self.crypto_validator.generate_key_pair(self.peer_id)
        self.peer_info.public_key = public_key
        
        # Start core components
        await self.connection_manager.start()
        
        bootstrap_nodes = self.config.bootstrap_nodes
        if not bootstrap_nodes:
            # Use default bootstrap nodes (these would be real addresses)
            bootstrap_nodes = [
                ("bootstrap1.steamdeck-p2p.net", 8001),
                ("bootstrap2.steamdeck-p2p.net", 8001),
            ]
        
        await self.dht.start(bootstrap_nodes)
        
        # Update peer info with actual listening address
        self.peer_info.address = "127.0.0.1"  # Would be actual IP
        self.peer_info.port = self.dht.port
        
        # Start background tasks
        asyncio.create_task(self._request_processor_loop())
        asyncio.create_task(self._cache_maintenance_loop())
        asyncio.create_task(self._community_learning_loop())
        asyncio.create_task(self._reputation_update_loop())
        
        # Announce ourselves to the network
        await self._announce_peer()
        
        logger.info("P2P shader distribution system started successfully")
    
    async def stop(self):
        """Stop P2P system"""
        logger.info("Stopping P2P shader distribution system")
        
        self.is_running = False
        
        # Save state before shutdown
        await self._save_state()
        
        # Stop components
        await self.dht.stop()
        # Connection manager would be stopped here
        
        logger.info("P2P shader distribution system stopped")
    
    async def request_shader(self, shader_hash: str, game_id: str, 
                           shader_type: ShaderType, 
                           priority: BandwidthPriority = BandwidthPriority.NORMAL,
                           callback: Callable = None) -> Optional[ShaderCacheEntry]:
        """Request shader from P2P network"""
        
        request = P2PShaderRequest(
            shader_hash=shader_hash,
            game_id=game_id,
            shader_type=shader_type,
            priority=priority,
            requestor_id=self.peer_id
        )
        
        self.stats['shaders_requested'] += 1
        
        # Check local cache first
        cache_key = request.get_cache_key()
        if cache_key in self.local_cache:
            cache_entry = self.local_cache[cache_key]
            self.cache_access_times[cache_key] = time.time()
            self.stats['shaders_found_locally'] += 1
            self.stats['cache_hits'] += 1
            
            logger.debug(f"Shader {shader_hash} found in local cache")
            
            # Update ML system with cache hit
            if self.ml_predictor:
                self.pattern_analyzer.record_shader_usage(
                    game_id, "local_cache", shader_hash, time.time()
                )
            
            return cache_entry
        
        self.stats['cache_misses'] += 1
        
        # Search P2P network
        logger.debug(f"Searching P2P network for shader {shader_hash}")
        
        try:
            # Use DHT to find shader
            cache_entry = await self.dht.lookup_shader_cache(shader_hash, game_id)
            
            if cache_entry:
                # Verify cache entry integrity
                if await self._verify_cache_entry(cache_entry):
                    # Store in local cache
                    await self._store_in_local_cache(cache_entry)
                    
                    self.stats['shaders_found_p2p'] += 1
                    self.stats['bytes_downloaded'] += cache_entry.compressed_size
                    
                    # Update reputation of source peer
                    if cache_entry.source_peer:
                        self.reputation_system.record_successful_transfer(
                            cache_entry.source_peer, cache_entry.compressed_size
                        )
                        self.reputation_system.record_validation_result(
                            cache_entry.source_peer, True
                        )
                    
                    # Contribute to community learning
                    if self.config.community_learning:
                        await self._contribute_usage_data(cache_entry, request)
                    
                    logger.info(f"Shader {shader_hash} found via P2P network")
                    return cache_entry
                else:
                    logger.warning(f"Shader {shader_hash} failed verification")
                    
                    # Update reputation negatively
                    if cache_entry.source_peer:
                        self.reputation_system.record_validation_result(
                            cache_entry.source_peer, False
                        )
        
        except Exception as e:
            logger.error(f"Error requesting shader {shader_hash}: {e}")
        
        # Store request for potential future fulfillment
        self.pending_requests[cache_key] = request
        if callback:
            self.request_callbacks[cache_key] = callback
        
        logger.debug(f"Shader {shader_hash} not found, request queued")
        return None
    
    async def share_shader(self, shader_data: bytes, shader_hash: str, 
                          game_id: str, shader_type: ShaderType,
                          metadata: Dict = None) -> bool:
        """Share shader in P2P network"""
        
        try:
            # Compress shader data
            compressed_data, compression_method = self.shader_compressor.compress_shader(
                shader_data, 
                CompressionMethod.LZMA if self.config.compression_enabled 
                else CompressionMethod.NONE
            )
            
            # Calculate checksum
            checksum = self.crypto_validator.calculate_checksum(shader_data)
            
            # Create cache entry
            cache_entry = ShaderCacheEntry(
                shader_hash=shader_hash,
                game_id=game_id,
                shader_type=shader_type,
                compressed_data=compressed_data,
                compression_method=compression_method,
                original_size=len(shader_data),
                compressed_size=len(compressed_data),
                checksum=checksum,
                creation_time=time.time(),
                source_peer=self.peer_id
            )
            
            # Sign cache entry if enabled
            if self.config.signature_verification:
                signature_data = (
                    shader_hash + game_id + checksum + self.peer_id
                ).encode()
                cache_entry.signature = self.crypto_validator.sign_data(
                    self.peer_id, signature_data
                )
            
            # Store in local cache
            await self._store_in_local_cache(cache_entry)
            
            # Store in DHT
            success = await self.dht.store_shader_cache(cache_entry)
            
            if success:
                self.stats['shaders_shared'] += 1
                self.stats['bytes_uploaded'] += len(compressed_data)
                
                # Update our reputation for sharing
                self.reputation_system.record_successful_transfer(
                    self.peer_id, len(compressed_data)
                )
                
                logger.info(f"Shared shader {shader_hash} in P2P network "
                           f"({cache_entry.get_compression_ratio():.1%} compression)")
                
                # Fulfill any pending requests
                await self._fulfill_pending_requests(cache_entry)
                
                return True
            else:
                logger.warning(f"Failed to store shader {shader_hash} in DHT")
                return False
                
        except Exception as e:
            logger.error(f"Error sharing shader {shader_hash}: {e}")
            return False
    
    async def prefetch_predicted_shaders(self, game_id: str, current_shader: str,
                                       count: int = 5) -> List[ShaderCacheEntry]:
        """Prefetch shaders based on ML predictions"""
        
        if not self.ml_predictor:
            return []
        
        # Get shader predictions from ML system
        predictions = self.pattern_analyzer.predict_next_shaders(
            game_id, current_shader, top_k=count
        )
        
        prefetched = []
        
        for shader_hash, probability in predictions:
            # Only prefetch high-probability shaders
            if probability < 0.5:
                continue
            
            # Determine shader type (would be enhanced with more metadata)
            shader_type = ShaderType.FRAGMENT  # Default, would be inferred
            
            # Request with low priority for prefetching
            cache_entry = await self.request_shader(
                shader_hash, game_id, shader_type, 
                BandwidthPriority.LOW
            )
            
            if cache_entry:
                prefetched.append(cache_entry)
                logger.debug(f"Prefetched shader {shader_hash} "
                           f"(probability: {probability:.2f})")
        
        logger.info(f"Prefetched {len(prefetched)}/{count} predicted shaders")
        return prefetched
    
    async def get_community_optimization_suggestions(self, game_id: str) -> List[Dict]:
        """Get optimization suggestions from community data"""
        
        if not self.config.community_learning:
            return []
        
        suggestions = []
        
        # Analyze community performance data
        if game_id in self.community_data['performance_metrics']:
            metrics = self.community_data['performance_metrics'][game_id]
            
            # Find common optimization patterns
            common_optimizations = defaultdict(int)
            for metric_data in metrics:
                for opt in metric_data.get('optimizations', []):
                    common_optimizations[opt] += 1
            
            # Convert to suggestions
            for optimization, count in common_optimizations.items():
                if count >= 3:  # Minimum consensus
                    suggestions.append({
                        'type': 'community_optimization',
                        'optimization': optimization,
                        'confidence': min(1.0, count / 10),  # 10 = max confidence
                        'game_id': game_id
                    })
        
        return suggestions
    
    def set_ml_predictor(self, predictor: SteamDeckShaderPredictor):
        """Set ML predictor integration"""
        self.ml_predictor = predictor
        logger.info("ML predictor integration enabled")
    
    async def _verify_cache_entry(self, cache_entry: ShaderCacheEntry) -> bool:
        """Verify cache entry integrity and signature"""
        
        try:
            # Decompress data
            decompressed_data = self.shader_compressor.decompress_shader(
                cache_entry.compressed_data, cache_entry.compression_method
            )
            
            # Verify checksum
            if not self.crypto_validator.verify_checksum(
                decompressed_data, cache_entry.checksum
            ):
                logger.warning(f"Checksum verification failed for {cache_entry.shader_hash}")
                return False
            
            # Verify signature if required
            if (self.config.signature_verification and 
                cache_entry.signature and cache_entry.source_peer):
                
                signature_data = (
                    cache_entry.shader_hash + cache_entry.game_id + 
                    cache_entry.checksum + cache_entry.source_peer
                ).encode()
                
                if not self.crypto_validator.verify_signature(
                    cache_entry.source_peer, signature_data, cache_entry.signature
                ):
                    logger.warning(f"Signature verification failed for {cache_entry.shader_hash}")
                    return False
            
            # Check source peer reputation if available
            if cache_entry.source_peer:
                reputation = self.reputation_system.get_reputation_score(cache_entry.source_peer)
                if reputation < self.config.reputation_threshold:
                    logger.warning(f"Source peer {cache_entry.source_peer} has low reputation: {reputation}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Cache entry verification error: {e}")
            return False
    
    async def _store_in_local_cache(self, cache_entry: ShaderCacheEntry):
        """Store cache entry in local cache"""
        
        cache_key = f"{cache_entry.game_id}:{cache_entry.shader_hash}"
        
        # Check cache size limit
        if (self.cache_size_bytes + cache_entry.compressed_size > 
            self.config.max_cache_size_mb * 1024 * 1024):
            
            # Make space by removing least recently used entries
            await self._cleanup_cache()
            
            # Check again after cleanup
            if (self.cache_size_bytes + cache_entry.compressed_size > 
                self.config.max_cache_size_mb * 1024 * 1024):
                logger.warning("Cache size limit exceeded, cannot store shader")
                return
        
        # Store entry
        self.local_cache[cache_key] = cache_entry
        self.cache_access_times[cache_key] = time.time()
        self.cache_size_bytes += cache_entry.compressed_size
        
        logger.debug(f"Stored shader {cache_entry.shader_hash} in local cache "
                    f"({self.cache_size_bytes / 1024 / 1024:.1f} MB used)")
    
    async def _cleanup_cache(self):
        """Clean up local cache using LRU policy"""
        
        # Sort by access time (least recent first)
        sorted_entries = sorted(
            self.cache_access_times.items(),
            key=lambda x: x[1]
        )
        
        # Remove oldest entries until we're under 80% of limit
        target_size = self.config.max_cache_size_mb * 1024 * 1024 * 0.8
        
        for cache_key, _ in sorted_entries:
            if self.cache_size_bytes <= target_size:
                break
            
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]
                self.cache_size_bytes -= entry.compressed_size
                del self.local_cache[cache_key]
                del self.cache_access_times[cache_key]
        
        logger.debug(f"Cache cleanup completed, {len(self.local_cache)} entries remaining")
    
    async def _fulfill_pending_requests(self, cache_entry: ShaderCacheEntry):
        """Fulfill pending requests with new cache entry"""
        
        cache_key = f"{cache_entry.game_id}:{cache_entry.shader_hash}"
        
        if cache_key in self.pending_requests:
            # Call callback if registered
            if cache_key in self.request_callbacks:
                callback = self.request_callbacks[cache_key]
                try:
                    await callback(cache_entry)
                except Exception as e:
                    logger.error(f"Callback error for {cache_key}: {e}")
                del self.request_callbacks[cache_key]
            
            del self.pending_requests[cache_key]
            logger.debug(f"Fulfilled pending request for {cache_entry.shader_hash}")
    
    async def _announce_peer(self):
        """Announce our presence to the network"""
        
        # Update reputation system with our Steam Deck verification
        self.reputation_system.set_steam_deck_verified(self.peer_id, True)
        
        # This would broadcast our peer info to connected peers
        logger.info(f"Announced peer {self.peer_id} to network")
    
    async def _contribute_usage_data(self, cache_entry: ShaderCacheEntry, 
                                   request: P2PShaderRequest):
        """Contribute usage data for community learning"""
        
        if not self.config.community_learning:
            return
        
        usage_data = {
            'shader_hash': cache_entry.shader_hash,
            'game_id': cache_entry.game_id,
            'shader_type': cache_entry.shader_type.value,
            'timestamp': time.time(),
            'requester': self.peer_id,
            'priority': request.priority.value,
            'cache_size': cache_entry.compressed_size,
            'compression_ratio': cache_entry.get_compression_ratio()
        }
        
        self.community_data['shader_usage_patterns'][cache_entry.game_id].append(usage_data)
        self.stats['community_contributions'] += 1
        
        logger.debug(f"Contributed usage data for {cache_entry.shader_hash}")
    
    async def _save_state(self):
        """Save system state to disk"""
        
        state_data = {
            'peer_id': self.peer_id,
            'stats': self.stats,
            'reputation_data': dict(self.reputation_system.peer_reputations),
            'community_data': dict(self.community_data),
            'config': asdict(self.config),
            'save_time': time.time()
        }
        
        state_file = Path("p2p_state.json")
        try:
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            logger.info(f"Saved P2P state to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    async def _load_state(self):
        """Load system state from disk"""
        
        state_file = Path("p2p_state.json")
        if not state_file.exists():
            return
        
        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore statistics
            if 'stats' in state_data:
                self.stats.update(state_data['stats'])
            
            # Restore reputation data
            if 'reputation_data' in state_data:
                for peer_id, rep_data in state_data['reputation_data'].items():
                    self.reputation_system.peer_reputations[peer_id] = rep_data
            
            # Restore community data
            if 'community_data' in state_data:
                for key, data in state_data['community_data'].items():
                    self.community_data[key].extend(data)
            
            logger.info(f"Loaded P2P state from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
    
    async def _request_processor_loop(self):
        """Process pending requests"""
        
        while self.is_running:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                
                current_time = time.time()
                expired_requests = []
                
                # Check for expired requests
                for cache_key, request in self.pending_requests.items():
                    if (current_time - request.timestamp) > request.max_wait_time:
                        expired_requests.append(cache_key)
                
                # Remove expired requests
                for cache_key in expired_requests:
                    del self.pending_requests[cache_key]
                    self.request_callbacks.pop(cache_key, None)
                
                if expired_requests:
                    logger.debug(f"Expired {len(expired_requests)} pending requests")
                
            except Exception as e:
                logger.error(f"Request processor loop error: {e}")
    
    async def _cache_maintenance_loop(self):
        """Periodic cache maintenance"""
        
        while self.is_running:
            try:
                await asyncio.sleep(self.config.cache_cleanup_interval)
                
                # Cleanup old cache entries
                await self._cleanup_cache()
                
                # Update cache statistics
                logger.info(f"Cache maintenance: {len(self.local_cache)} entries, "
                           f"{self.cache_size_bytes / 1024 / 1024:.1f} MB")
                
            except Exception as e:
                logger.error(f"Cache maintenance loop error: {e}")
    
    async def _community_learning_loop(self):
        """Process community learning data"""
        
        while self.is_running:
            try:
                await asyncio.sleep(600)  # Process every 10 minutes
                
                if not self.config.community_learning:
                    continue
                
                # Analyze usage patterns for insights
                for game_id, usage_data in self.community_data['shader_usage_patterns'].items():
                    if len(usage_data) < 10:  # Need minimum data for analysis
                        continue
                    
                    # Find common optimization patterns
                    # This would implement more sophisticated analysis
                    logger.debug(f"Analyzing {len(usage_data)} usage patterns for {game_id}")
                
            except Exception as e:
                logger.error(f"Community learning loop error: {e}")
    
    async def _reputation_update_loop(self):
        """Update peer reputations"""
        
        while self.is_running:
            try:
                await asyncio.sleep(1800)  # Update every 30 minutes
                
                # Clean up old reputation entries
                self.reputation_system.cleanup_old_entries()
                
                # Update our own reputation based on contributions
                contribution_score = self.stats['shaders_shared'] + self.stats['community_contributions']
                self.reputation_system.peer_reputations[self.peer_id]['contribution_score'] = contribution_score
                
                logger.debug("Updated peer reputations")
                
            except Exception as e:
                logger.error(f"Reputation update loop error: {e}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        
        uptime_hours = (time.time() - self.stats['start_time']) / 3600
        
        return {
            'peer_id': self.peer_id,
            'uptime_hours': uptime_hours,
            'is_running': self.is_running,
            'cache_stats': {
                'local_entries': len(self.local_cache),
                'cache_size_mb': self.cache_size_bytes / 1024 / 1024,
                'cache_limit_mb': self.config.max_cache_size_mb,
                'cache_utilization': (self.cache_size_bytes / 
                                    (self.config.max_cache_size_mb * 1024 * 1024))
            },
            'network_stats': self.stats.copy(),
            'dht_stats': self.dht.get_stats(),
            'connection_stats': self.connection_manager.get_stats(),
            'reputation_stats': {
                'tracked_peers': len(self.reputation_system.peer_reputations),
                'blacklisted_peers': len(self.reputation_system.blacklisted_peers),
                'our_reputation': self.reputation_system.get_reputation_score(self.peer_id)
            },
            'community_stats': {
                'usage_patterns': sum(len(data) for data in 
                                    self.community_data['shader_usage_patterns'].values()),
                'performance_metrics': sum(len(data) for data in 
                                         self.community_data['performance_metrics'].values()),
                'contributions': self.stats['community_contributions']
            },
            'pending_requests': len(self.pending_requests)
        }


class IntegratedShaderSystem:
    """Complete integrated system combining ML prediction and P2P distribution"""
    
    def __init__(self, peer_id: str, config: P2PConfig = None):
        self.peer_id = peer_id
        
        # Initialize ML prediction system
        self.ml_system = SteamDeckOptimizedSystem()
        
        # Initialize P2P distribution system
        self.p2p_system = P2PShaderDistributionSystem(peer_id, config)
        
        # Integration state
        self.active_game_id = None
        self.prediction_cache = {}  # Cache ML predictions
        
        logger.info(f"Integrated shader system initialized for peer {peer_id}")
    
    async def start(self):
        """Start integrated system"""
        logger.info("Starting integrated shader system")
        
        # Start ML system
        self.ml_system.start()
        
        # Start P2P system with ML integration
        await self.p2p_system.start(self.ml_system.predictor)
        self.p2p_system.set_ml_predictor(self.ml_system.predictor)
        
        logger.info("Integrated shader system started")
    
    async def stop(self):
        """Stop integrated system"""
        logger.info("Stopping integrated shader system")
        
        await self.p2p_system.stop()
        self.ml_system.stop()
        
        logger.info("Integrated shader system stopped")
    
    def set_active_game(self, game_id: str, game_name: str):
        """Set active game for both systems"""
        self.active_game_id = game_id
        self.ml_system.set_active_game(game_id, game_name)
        logger.info(f"Active game set to {game_name} (ID: {game_id})")
    
    async def request_shader_intelligent(self, shader_data: Dict) -> Dict:
        """Intelligent shader request combining ML prediction and P2P"""
        
        shader_hash = shader_data.get('hash', '')
        game_id = shader_data.get('game_id', self.active_game_id)
        shader_type = ShaderType(shader_data.get('type', 'fragment'))
        
        # Get ML prediction first
        ml_result = self.ml_system.process_shader(shader_data)
        
        # Determine request priority based on ML prediction
        if ml_result.get('can_compile_now', False):
            # If we can compile immediately, lower P2P priority
            priority = BandwidthPriority.LOW
        else:
            # If compilation is delayed, higher P2P priority
            thermal_state = ml_result.get('thermal_state', 'normal')
            if thermal_state in ['hot', 'throttling']:
                priority = BandwidthPriority.HIGH
            else:
                priority = BandwidthPriority.NORMAL
        
        # Try P2P network first
        start_time = time.time()
        cache_entry = await self.p2p_system.request_shader(
            shader_hash, game_id, shader_type, priority
        )
        
        p2p_time = time.time() - start_time
        
        if cache_entry:
            # Found via P2P - avoid compilation
            result = {
                'source': 'p2p',
                'cache_entry': cache_entry,
                'predicted_compilation_time_ms': ml_result.get('predicted_compilation_time_ms', 0),
                'actual_retrieval_time_ms': p2p_time * 1000,
                'thermal_state': ml_result.get('thermal_state'),
                'time_saved_ms': ml_result.get('predicted_compilation_time_ms', 0) - (p2p_time * 1000),
                'compression_ratio': cache_entry.get_compression_ratio(),
                'source_peer': cache_entry.source_peer
            }
            
            logger.info(f"Shader {shader_hash} retrieved via P2P "
                       f"(saved {result['time_saved_ms']:.1f}ms compilation time)")
            
            return result
        
        else:
            # Not found in P2P, use ML prediction for compilation decision
            result = ml_result.copy()
            result['source'] = 'compilation'
            result['p2p_search_time_ms'] = p2p_time * 1000
            
            # Prefetch related shaders based on ML predictions
            if game_id:
                asyncio.create_task(self._prefetch_related_shaders(game_id, shader_hash))
            
            return result
    
    async def share_compiled_shader(self, shader_data: Dict, compilation_result: Dict) -> bool:
        """Share newly compiled shader in P2P network"""
        
        # Extract shader information
        shader_hash = shader_data.get('hash', '')
        game_id = shader_data.get('game_id', self.active_game_id)
        shader_type = ShaderType(shader_data.get('type', 'fragment'))
        
        # Get compiled shader bytecode (simulated)
        bytecode = compilation_result.get('bytecode', b'simulated_bytecode')
        
        # Share in P2P network
        success = await self.p2p_system.share_shader(
            bytecode, shader_hash, game_id, shader_type,
            metadata={
                'compilation_time_ms': compilation_result.get('time_ms', 0),
                'gpu_temp': compilation_result.get('gpu_temp', 0),
                'power_draw': compilation_result.get('power_draw', 0)
            }
        )
        
        # Record compilation result in ML system
        self.ml_system.record_compilation_result(shader_data, compilation_result)
        
        return success
    
    async def _prefetch_related_shaders(self, game_id: str, current_shader: str):
        """Prefetch related shaders based on ML predictions"""
        
        try:
            prefetched = await self.p2p_system.prefetch_predicted_shaders(
                game_id, current_shader, count=3
            )
            
            if prefetched:
                logger.info(f"Prefetched {len(prefetched)} related shaders for {game_id}")
            
        except Exception as e:
            logger.error(f"Prefetch error: {e}")
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics from all systems"""
        
        return {
            'integrated_system': {
                'peer_id': self.peer_id,
                'active_game': self.active_game_id,
                'ml_enabled': True,
                'p2p_enabled': True
            },
            'ml_system': self.ml_system.get_system_status(),
            'p2p_system': self.p2p_system.get_stats()
        }