//! P2P shader distribution network with QUIC transport and reputation system
//! 
//! This module implements a peer-to-peer network for distributing shader caches
//! with cryptographic validation, bandwidth optimization, and Steam Deck awareness.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;

pub mod transport;
pub mod dht;
pub mod reputation;
pub mod bandwidth;
pub mod crypto;

// Re-export main types
pub use transport::{QuicTransport, PeerConnection};
pub use dht::{DistributedHashTable, DhtNode};
pub use reputation::{ReputationSystem, PeerReputation};
pub use bandwidth::{BandwidthManager, BandwidthProfile};
pub use crypto::{CryptoManager, PeerIdentity};

/// Main P2P network manager
pub struct P2PNetwork {
    transport: Arc<QuicTransport>,
    dht: Arc<DistributedHashTable>,
    reputation: Arc<ReputationSystem>,
    bandwidth: Arc<BandwidthManager>,
    crypto: Arc<CryptoManager>,
    local_peer_id: PeerIdentity,
}

/// Shader cache entry for P2P distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct P2PShaderCache {
    pub shader_hash: String,
    pub game_id: String,
    pub bytecode: Vec<u8>,
    pub metadata: ShaderMetadata,
    pub signature: Vec<u8>,
    pub peer_id: PeerIdentity,
    pub timestamp: u64,
}

/// Metadata for shader cache entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaderMetadata {
    pub compilation_time_ms: f32,
    pub shader_type: String,
    pub hardware_compatibility: Vec<String>,
    pub optimization_level: u8,
    pub compressed_size: usize,
    pub compression_ratio: f32,
}

impl P2PNetwork {
    /// Create a new P2P network
    pub fn new() -> Result<Self> {
        let crypto = Arc::new(CryptoManager::new()?);
        let local_peer_id = crypto.generate_peer_identity()?;
        let transport = Arc::new(QuicTransport::new(local_peer_id.clone())?);
        let dht = Arc::new(DistributedHashTable::new(local_peer_id.clone())?);
        let reputation = Arc::new(ReputationSystem::new()?);
        let bandwidth = Arc::new(BandwidthManager::new_steam_deck()?);
        
        Ok(Self {
            transport,
            dht,
            reputation,
            bandwidth,
            crypto,
            local_peer_id,
        })
    }
    
    /// Start the P2P network
    pub async fn start(&self, listen_port: u16) -> Result<()> {
        // Start transport layer
        self.transport.start_listening(listen_port).await?;
        
        // Bootstrap DHT with known nodes
        self.dht.bootstrap().await?;
        
        // Start reputation monitoring
        self.reputation.start_monitoring().await?;
        
        tracing::info!("P2P network started on port {}", listen_port);
        Ok(())
    }
    
    /// Publish shader cache to the network
    pub async fn publish_shader(&self, cache: P2PShaderCache) -> Result<()> {
        // Validate and sign the shader cache
        let signed_cache = self.crypto.sign_shader_cache(cache)?;
        
        // Store in DHT
        self.dht.store_shader(signed_cache.shader_hash.clone(), signed_cache.clone()).await?;
        
        // Replicate to nearby peers
        let peers = self.dht.find_closest_peers(&signed_cache.shader_hash, 3).await?;
        for peer in peers {
            if let Ok(conn) = self.transport.connect_to_peer(peer).await {
                let _ = conn.send_shader_cache(signed_cache.clone()).await;
            }
        }
        
        tracing::info!("Published shader cache: {}", signed_cache.shader_hash);
        Ok(())
    }
    
    /// Request shader cache from the network
    pub async fn request_shader(&self, shader_hash: &str) -> Result<Option<P2PShaderCache>> {
        // Check local DHT first
        if let Some(cache) = self.dht.get_shader(shader_hash).await? {
            return Ok(Some(cache));
        }
        
        // Find peers that might have the shader
        let peers = self.dht.find_closest_peers(shader_hash, 5).await?;
        
        for peer in peers {
            // Check peer reputation
            let reputation = self.reputation.get_peer_reputation(&peer).await?;
            if reputation.trust_score < 0.5 {
                continue; // Skip untrusted peers
            }
            
            // Request shader from peer
            if let Ok(conn) = self.transport.connect_to_peer(peer).await {
                if let Ok(Some(cache)) = conn.request_shader_cache(shader_hash).await {
                    // Verify signature
                    if self.crypto.verify_shader_cache(&cache)? {
                        // Update peer reputation (positive)
                        self.reputation.record_successful_transfer(&cache.peer_id).await?;
                        return Ok(Some(cache));
                    } else {
                        // Update peer reputation (negative)
                        self.reputation.record_failed_transfer(&cache.peer_id).await?;
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    /// Get network statistics
    pub async fn get_network_stats(&self) -> Result<NetworkStats> {
        let dht_stats = self.dht.get_stats().await?;
        let bandwidth_stats = self.bandwidth.get_stats().await?;
        let reputation_stats = self.reputation.get_stats().await?;
        
        Ok(NetworkStats {
            connected_peers: dht_stats.peer_count,
            stored_shaders: dht_stats.stored_entries,
            bandwidth_usage_mbps: bandwidth_stats.current_usage_mbps,
            trusted_peers: reputation_stats.trusted_peer_count,
            average_trust_score: reputation_stats.average_trust_score,
        })
    }
    
    /// Handle incoming peer connections
    pub async fn handle_peer_connection(&self, peer_id: PeerIdentity) -> Result<()> {
        // Update DHT routing table
        self.dht.add_peer(peer_id.clone()).await?;
        
        // Initialize reputation tracking
        self.reputation.initialize_peer_reputation(peer_id).await?;
        
        Ok(())
    }
    
    /// Optimize bandwidth usage for Steam Deck
    pub async fn optimize_for_steam_deck(&self) -> Result<()> {
        // Adjust bandwidth limits based on network conditions
        let profile = BandwidthProfile::steam_deck_wifi();
        self.bandwidth.apply_profile(profile).await?;
        
        // Prioritize shader requests over publishing
        self.bandwidth.set_priority_weights(0.7, 0.3).await?; // 70% download, 30% upload
        
        Ok(())
    }
}

/// Network performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub connected_peers: usize,
    pub stored_shaders: usize,
    pub bandwidth_usage_mbps: f32,
    pub trusted_peers: usize,
    pub average_trust_score: f32,
}

/// Error types for P2P networking
#[derive(thiserror::Error, Debug)]
pub enum P2PError {
    #[error("Transport error: {0}")]
    TransportError(String),
    
    #[error("DHT error: {0}")]
    DhtError(String),
    
    #[error("Reputation error: {0}")]
    ReputationError(String),
    
    #[error("Bandwidth error: {0}")]
    BandwidthError(String),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Protocol error: {0}")]
    ProtocolError(String),
}