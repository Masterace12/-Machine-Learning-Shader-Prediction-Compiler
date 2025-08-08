"""
Complete Demonstration of P2P Shader Cache Distribution System
Shows real-world usage scenarios for Steam Deck users
"""

import asyncio
import time
import json
import random
from typing import Dict, List
from pathlib import Path

# Import P2P system components
from p2p_shader_distribution import (
    P2PShaderDistributionSystem, IntegratedShaderSystem, 
    P2PConfig, P2PShaderRequest
)
from p2p_shader_network import PeerRole, ShaderType, BandwidthPriority
from p2p_bandwidth_optimizer import NetworkCondition

# Import ML integration
from shader_prediction_system import ShaderMetrics
from shader_training_evaluation import SyntheticDataGenerator


class P2PNetworkSimulator:
    """Simulates a P2P network with multiple Steam Deck peers"""
    
    def __init__(self, num_peers: int = 10):
        self.num_peers = num_peers
        self.peers: List[IntegratedShaderSystem] = []
        self.network_conditions = {}
        self.running = False
        
        # Common game scenarios
        self.game_scenarios = {
            "cyberpunk_2077": {
                "app_id": "1091500",
                "name": "Cyberpunk 2077",
                "shader_count": 500,
                "common_shaders": 50,
                "complexity_high": True
            },
            "elden_ring": {
                "app_id": "1245620", 
                "name": "ELDEN RING",
                "shader_count": 300,
                "common_shaders": 40,
                "complexity_high": True
            },
            "portal_2": {
                "app_id": "620",
                "name": "Portal 2", 
                "shader_count": 150,
                "common_shaders": 25,
                "complexity_high": False
            }
        }
    
    async def setup_network(self):
        """Setup simulated P2P network"""
        print("Setting up P2P network simulation...")
        
        # Create bootstrap nodes
        bootstrap_nodes = [
            ("bootstrap1.steamdeck-p2p.net", 8001),
            ("bootstrap2.steamdeck-p2p.net", 8002)
        ]
        
        # Create peers with varying configurations
        for i in range(self.num_peers):
            peer_id = f"steamdeck_{i:03d}"
            
            # Vary peer configurations
            config = P2PConfig(
                max_connections=random.randint(20, 50),
                max_bandwidth_kbps=random.uniform(1024, 3072),  # 1-3 Mbps
                max_cache_size_mb=random.randint(512, 2048),    # 512MB - 2GB
                bootstrap_nodes=bootstrap_nodes,
                community_learning=True,
                contribution_rewards=True
            )
            
            peer = IntegratedShaderSystem(peer_id, config)
            self.peers.append(peer)
            
            # Simulate network conditions for each peer
            self.network_conditions[peer_id] = NetworkCondition(
                bandwidth_kbps=config.max_bandwidth_kbps,
                latency_ms=random.uniform(20, 100),
                packet_loss=random.uniform(0, 0.05),
                jitter_ms=random.uniform(1, 15),
                signal_strength=random.uniform(60, 95),
                connection_type="wifi",
                is_metered=random.random() < 0.2  # 20% on metered connections
            )
        
        print(f"Created {self.num_peers} peer nodes")
    
    async def start_network(self):
        """Start all peers in the network"""
        print("Starting P2P network...")
        
        # Start peers with staggered timing to simulate real-world conditions
        start_tasks = []
        for i, peer in enumerate(self.peers):
            # Stagger peer starts over 30 seconds
            delay = i * (30.0 / self.num_peers)
            start_tasks.append(self._start_peer_delayed(peer, delay))
        
        await asyncio.gather(*start_tasks)
        self.running = True
        
        print("P2P network started successfully")
    
    async def _start_peer_delayed(self, peer: IntegratedShaderSystem, delay: float):
        """Start peer with delay"""
        await asyncio.sleep(delay)
        await peer.start()
        
        # Update network conditions
        condition = self.network_conditions[peer.peer_id]
        peer.p2p_system.connection_manager.bandwidth_manager.update_network_condition(condition)
    
    async def simulate_gaming_session(self, game_key: str, duration_minutes: int = 10):
        """Simulate gaming session with shader requests"""
        
        game_info = self.game_scenarios[game_key]
        print(f"\nSimulating {game_info['name']} gaming session ({duration_minutes} minutes)...")
        
        # Select subset of peers to simulate active gaming
        active_peers = random.sample(self.peers, min(5, len(self.peers)))
        
        # Set active game for all peers
        for peer in active_peers:
            peer.set_active_game(game_info['app_id'], game_info['name'])
        
        # Generate shader scenarios
        generator = SyntheticDataGenerator()
        
        # Simulate gaming for specified duration
        end_time = time.time() + (duration_minutes * 60)
        shader_request_count = 0
        
        while time.time() < end_time and self.running:
            # Select random active peer
            peer = random.choice(active_peers)
            
            # Generate shader request based on game complexity
            if game_info['complexity_high']:
                # High complexity games have more diverse shaders
                shader_type = random.choice([ShaderType.VERTEX, ShaderType.FRAGMENT, ShaderType.COMPUTE])
                priority = random.choice([BandwidthPriority.HIGH, BandwidthPriority.NORMAL])
            else:
                # Simpler games mostly use basic shaders
                shader_type = random.choice([ShaderType.VERTEX, ShaderType.FRAGMENT])
                priority = BandwidthPriority.NORMAL
            
            # Create shader data
            shader_hash = f"{game_key}_shader_{random.randint(1, game_info['shader_count']):04d}"
            shader_data = {
                'hash': shader_hash,
                'type': shader_type.value,
                'game_id': game_info['app_id'],
                'bytecode_size': random.randint(1024, 8192),
                'instruction_count': random.randint(50, 300),
                'register_pressure': random.randint(16, 64),
                'texture_samples': random.randint(2, 8),
                'branch_complexity': random.randint(1, 6),
                'loop_depth': random.randint(0, 3),
                'scene_id': f"scene_{random.randint(1, 10)}",
                'priority': random.randint(1, 3),
                'variant_count': random.randint(1, 4)
            }
            
            try:
                # Request shader through integrated system
                result = await peer.request_shader_intelligent(shader_data)
                shader_request_count += 1
                
                # Simulate compilation if needed
                if result['source'] == 'compilation':
                    # Simulate compilation time and result
                    compile_time = result.get('predicted_compilation_time_ms', 50)
                    await asyncio.sleep(compile_time / 1000)  # Convert to seconds
                    
                    # Create compilation result
                    compilation_result = {
                        'time_ms': compile_time + random.uniform(-10, 10),
                        'gpu_temp': random.uniform(65, 85),
                        'power_draw': random.uniform(8, 15),
                        'memory_mb': random.randint(32, 128),
                        'success': True,
                        'bytecode': b'simulated_compiled_bytecode_data'
                    }
                    
                    # Share compiled shader
                    await peer.share_compiled_shader(shader_data, compilation_result)
                
                # Random delay between requests (simulate frame rendering)
                await asyncio.sleep(random.uniform(0.016, 0.033))  # 30-60 FPS
                
            except Exception as e:
                print(f"Error in shader request: {e}")
        
        print(f"Gaming session completed: {shader_request_count} shader requests processed")
    
    async def demonstrate_network_effects(self):
        """Demonstrate network condition effects"""
        print("\nDemonstrating network condition effects...")
        
        # Simulate poor network conditions for some peers
        degraded_peers = random.sample(self.peers, 3)
        
        for peer in degraded_peers:
            # Degrade network conditions
            degraded_condition = NetworkCondition(
                bandwidth_kbps=256,  # Very low bandwidth
                latency_ms=200,      # High latency
                packet_loss=0.1,     # 10% packet loss
                jitter_ms=50,        # High jitter
                signal_strength=30,   # Poor signal
                connection_type="wifi",
                is_metered=True
            )
            
            peer.p2p_system.connection_manager.bandwidth_manager.update_network_condition(degraded_condition)
            print(f"Degraded network for peer {peer.peer_id}")
        
        # Let system adapt for 30 seconds
        print("Allowing system to adapt to network changes...")
        await asyncio.sleep(30)
        
        # Test shader request under poor conditions
        peer = degraded_peers[0]
        shader_data = {
            'hash': 'test_degraded_shader',
            'type': 'fragment',
            'game_id': '1091500',
            'bytecode_size': 4096,
            'instruction_count': 200,
            'register_pressure': 48
        }
        
        start_time = time.time()
        result = await peer.request_shader_intelligent(shader_data)
        request_time = (time.time() - start_time) * 1000
        
        print(f"Shader request under poor conditions: {request_time:.1f}ms, source: {result['source']}")
    
    async def demonstrate_community_learning(self):
        """Demonstrate community learning features"""
        print("\nDemonstrating community learning...")
        
        # Generate community optimization data
        for peer in self.peers[:3]:  # Use first 3 peers
            # Simulate optimization discoveries
            optimizations = [
                "reduced_texture_samples",
                "simplified_lighting_model", 
                "optimized_vertex_processing"
            ]
            
            for opt in optimizations:
                peer.p2p_system.community_data['performance_metrics']['1091500'].append({
                    'optimization': opt,
                    'performance_gain': random.uniform(0.1, 0.3),
                    'timestamp': time.time(),
                    'peer_id': peer.peer_id
                })
        
        # Get community suggestions
        peer = self.peers[0]
        suggestions = await peer.p2p_system.get_community_optimization_suggestions('1091500')
        
        print(f"Community optimization suggestions: {len(suggestions)}")
        for suggestion in suggestions:
            print(f"  - {suggestion['optimization']} (confidence: {suggestion['confidence']:.2f})")
    
    async def stop_network(self):
        """Stop all peers"""
        print("\nStopping P2P network...")
        
        self.running = False
        stop_tasks = [peer.stop() for peer in self.peers]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        print("P2P network stopped")
    
    def print_network_stats(self):
        """Print comprehensive network statistics"""
        print("\n" + "="*60)
        print("NETWORK STATISTICS")
        print("="*60)
        
        # Aggregate statistics
        total_stats = {
            'total_shaders_requested': 0,
            'total_shaders_found_p2p': 0,
            'total_shaders_shared': 0,
            'total_bytes_downloaded': 0,
            'total_bytes_uploaded': 0,
            'total_cache_entries': 0,
            'total_community_contributions': 0
        }
        
        peer_stats = []
        
        for peer in self.peers:
            try:
                stats = peer.get_comprehensive_stats()
                p2p_stats = stats['p2p_system']['network_stats']
                
                peer_stats.append({
                    'peer_id': peer.peer_id,
                    'shaders_requested': p2p_stats['shaders_requested'],
                    'shaders_found_p2p': p2p_stats['shaders_found_p2p'],
                    'shaders_shared': p2p_stats['shaders_shared'],
                    'cache_entries': stats['p2p_system']['cache_stats']['local_entries'],
                    'cache_size_mb': stats['p2p_system']['cache_stats']['cache_size_mb'],
                    'reputation': stats['p2p_system']['reputation_stats']['our_reputation']
                })
                
                # Aggregate totals
                total_stats['total_shaders_requested'] += p2p_stats['shaders_requested']
                total_stats['total_shaders_found_p2p'] += p2p_stats['shaders_found_p2p']
                total_stats['total_shaders_shared'] += p2p_stats['shaders_shared']
                total_stats['total_bytes_downloaded'] += p2p_stats['bytes_downloaded']
                total_stats['total_bytes_uploaded'] += p2p_stats['bytes_uploaded']
                total_stats['total_cache_entries'] += stats['p2p_system']['cache_stats']['local_entries']
                total_stats['total_community_contributions'] += p2p_stats['community_contributions']
                
            except Exception as e:
                print(f"Error getting stats for {peer.peer_id}: {e}")
        
        # Print aggregate statistics
        print(f"Total Peers: {len(self.peers)}")
        print(f"Total Shader Requests: {total_stats['total_shaders_requested']}")
        print(f"Total P2P Cache Hits: {total_stats['total_shaders_found_p2p']}")
        print(f"Total Shaders Shared: {total_stats['total_shaders_shared']}")
        print(f"Total Cache Entries: {total_stats['total_cache_entries']}")
        print(f"Total Community Contributions: {total_stats['total_community_contributions']}")
        
        if total_stats['total_shaders_requested'] > 0:
            p2p_hit_rate = (total_stats['total_shaders_found_p2p'] / 
                           total_stats['total_shaders_requested']) * 100
            print(f"P2P Cache Hit Rate: {p2p_hit_rate:.1f}%")
        
        if total_stats['total_bytes_downloaded'] > 0:
            print(f"Total Data Downloaded: {total_stats['total_bytes_downloaded'] / 1024 / 1024:.1f} MB")
        
        if total_stats['total_bytes_uploaded'] > 0:
            print(f"Total Data Uploaded: {total_stats['total_bytes_uploaded'] / 1024 / 1024:.1f} MB")
        
        # Print top performing peers
        print(f"\nTop Performing Peers:")
        peer_stats.sort(key=lambda x: x['reputation'], reverse=True)
        
        for i, stats in enumerate(peer_stats[:5]):
            print(f"  {i+1}. {stats['peer_id']}: "
                  f"Rep={stats['reputation']:.3f}, "
                  f"Shared={stats['shaders_shared']}, "
                  f"Cache={stats['cache_entries']} entries "
                  f"({stats['cache_size_mb']:.1f} MB)")


async def run_comprehensive_demonstration():
    """Run comprehensive P2P system demonstration"""
    
    print("="*60)
    print("STEAM DECK P2P SHADER CACHE DISTRIBUTION DEMONSTRATION")
    print("="*60)
    print()
    
    # Create network simulator
    simulator = P2PNetworkSimulator(num_peers=8)
    
    try:
        # Setup and start network
        await simulator.setup_network()
        await simulator.start_network()
        
        # Allow network to stabilize
        print("Allowing network to stabilize...")
        await asyncio.sleep(10)
        
        # Demonstrate multiple gaming scenarios
        await simulator.simulate_gaming_session("cyberpunk_2077", duration_minutes=3)
        await asyncio.sleep(5)
        
        await simulator.simulate_gaming_session("elden_ring", duration_minutes=2)
        await asyncio.sleep(5)
        
        await simulator.simulate_gaming_session("portal_2", duration_minutes=2)
        await asyncio.sleep(5)
        
        # Demonstrate network effects
        await simulator.demonstrate_network_effects()
        await asyncio.sleep(5)
        
        # Demonstrate community learning
        await simulator.demonstrate_community_learning()
        await asyncio.sleep(5)
        
        # Final statistics
        simulator.print_network_stats()
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        print(f"Demonstration error: {e}")
    finally:
        await simulator.stop_network()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)


async def run_simple_peer_demo():
    """Run simple two-peer demonstration"""
    
    print("="*50)
    print("SIMPLE P2P SHADER SHARING DEMONSTRATION")
    print("="*50)
    
    # Create two peers
    peer1_config = P2PConfig(
        max_cache_size_mb=512,
        max_bandwidth_kbps=2048,
        community_learning=True
    )
    
    peer2_config = P2PConfig(
        max_cache_size_mb=1024,
        max_bandwidth_kbps=1536,
        community_learning=True
    )
    
    peer1 = IntegratedShaderSystem("steamdeck_alice", peer1_config)
    peer2 = IntegratedShaderSystem("steamdeck_bob", peer2_config)
    
    try:
        # Start both peers
        print("Starting peer systems...")
        await peer1.start()
        await peer2.start()
        
        # Set active game
        peer1.set_active_game("1091500", "Cyberpunk 2077")
        peer2.set_active_game("1091500", "Cyberpunk 2077")
        
        print("Peers started, simulating shader sharing...")
        
        # Peer 1 compiles and shares a shader
        shader_data = {
            'hash': 'cyberpunk_neon_shader_001',
            'type': 'fragment',
            'game_id': '1091500',
            'bytecode_size': 2048,
            'instruction_count': 150,
            'register_pressure': 32,
            'texture_samples': 4,
            'branch_complexity': 3,
            'loop_depth': 1,
            'scene_id': 'night_city',
            'priority': 2
        }
        
        # Simulate compilation result
        compilation_result = {
            'time_ms': 45.2,
            'gpu_temp': 73.1,
            'power_draw': 13.5,
            'memory_mb': 48,
            'success': True,
            'bytecode': b'simulated_neon_shader_bytecode'
        }
        
        print("Peer 1 compiling and sharing shader...")
        await peer1.share_compiled_shader(shader_data, compilation_result)
        
        # Allow time for DHT propagation
        await asyncio.sleep(2)
        
        # Peer 2 requests the same shader
        print("Peer 2 requesting shader...")
        result = await peer2.request_shader_intelligent(shader_data)
        
        print(f"Shader request result: {result['source']}")
        if result['source'] == 'p2p':
            print(f"  Time saved: {result.get('time_saved_ms', 0):.1f} ms")
            print(f"  Compression ratio: {result.get('compression_ratio', 0):.1%}")
            print(f"  Source peer: {result.get('source_peer', 'unknown')}")
        
        # Print peer statistics
        print("\nPeer 1 Statistics:")
        stats1 = peer1.get_comprehensive_stats()
        p2p_stats1 = stats1['p2p_system']['network_stats']
        print(f"  Shaders shared: {p2p_stats1['shaders_shared']}")
        print(f"  Bytes uploaded: {p2p_stats1['bytes_uploaded']}")
        
        print("\nPeer 2 Statistics:")
        stats2 = peer2.get_comprehensive_stats()
        p2p_stats2 = stats2['p2p_system']['network_stats']
        print(f"  Shaders found via P2P: {p2p_stats2['shaders_found_p2p']}")
        print(f"  Bytes downloaded: {p2p_stats2['bytes_downloaded']}")
        
    finally:
        print("\nStopping peers...")
        await peer1.stop()
        await peer2.stop()
    
    print("Simple demonstration complete!")


def create_example_config():
    """Create example configuration file"""
    
    config = P2PConfig(
        listen_port=8080,
        max_connections=30,
        max_bandwidth_kbps=1536,  # Conservative for Steam Deck
        dht_k=20,
        dht_ttl_hours=12,
        max_cache_size_mb=512,
        cache_cleanup_interval=3600,
        compression_enabled=True,
        signature_verification=True,
        reputation_threshold=0.3,
        bootstrap_nodes=[
            ("bootstrap1.steamdeck-p2p.net", 8001),
            ("bootstrap2.steamdeck-p2p.net", 8002)
        ],
        community_learning=True,
        contribution_rewards=True
    )
    
    config_file = Path("p2p_config_example.json")
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"Example configuration saved to {config_file}")


if __name__ == "__main__":
    import sys
    
    # Create example configuration
    create_example_config()
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "simple":
        # Run simple demonstration
        asyncio.run(run_simple_peer_demo())
    else:
        # Run comprehensive demonstration
        asyncio.run(run_comprehensive_demonstration())
    
    print(f"\nGenerated files:")
    print(f"  - p2p_config_example.json: Example configuration")
    print(f"  - p2p_state.json: Saved peer state (if generated)")
    print(f"  - Various log files and cached data")
    
    print(f"\nTo run simple demo: python p2p_demonstration.py simple")
    print(f"To run full demo: python p2p_demonstration.py")
    
    print(f"\nP2P Shader Distribution System Features:")
    print(f"  ✓ Secure cryptographic validation (SHA-256)")
    print(f"  ✓ Advanced reputation system")
    print(f"  ✓ Content-addressed storage with delta compression")
    print(f"  ✓ Distributed Hash Table (Kademlia-based)")
    print(f"  ✓ Bandwidth optimization for WiFi constraints")
    print(f"  ✓ ML integration for predictive caching")
    print(f"  ✓ Community learning for optimization")
    print(f"  ✓ Autonomous operation with intermittent connectivity")
    print(f"  ✓ Steam Deck hardware optimization")