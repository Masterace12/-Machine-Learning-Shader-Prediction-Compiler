//! Peer reputation system for trust management

use anyhow::Result;
use std::collections::HashMap;
use crate::crypto::PeerIdentity;

/// Reputation system
pub struct ReputationSystem {
    reputations: HashMap<PeerIdentity, PeerReputation>,
}

/// Peer reputation data
#[derive(Debug, Clone)]
pub struct PeerReputation {
    pub trust_score: f32,
    pub successful_transfers: u32,
    pub failed_transfers: u32,
    pub last_interaction: u64,
}

/// Reputation statistics
pub struct ReputationStats {
    pub trusted_peer_count: usize,
    pub average_trust_score: f32,
}

impl ReputationSystem {
    /// Create new reputation system
    pub fn new() -> Result<Self> {
        Ok(Self {
            reputations: HashMap::new(),
        })
    }
    
    /// Start reputation monitoring
    pub async fn start_monitoring(&self) -> Result<()> {
        // Stub implementation
        tracing::info!("Reputation monitoring started");
        Ok(())
    }
    
    /// Get peer reputation
    pub async fn get_peer_reputation(&self, peer_id: &PeerIdentity) -> Result<PeerReputation> {
        Ok(self.reputations.get(peer_id).cloned().unwrap_or_else(|| PeerReputation {
            trust_score: 0.5, // Default neutral trust
            successful_transfers: 0,
            failed_transfers: 0,
            last_interaction: 0,
        }))
    }
    
    /// Initialize reputation for new peer
    pub async fn initialize_peer_reputation(&mut self, peer_id: PeerIdentity) -> Result<()> {
        self.reputations.entry(peer_id).or_insert(PeerReputation {
            trust_score: 0.5,
            successful_transfers: 0,
            failed_transfers: 0,
            last_interaction: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });
        Ok(())
    }
    
    /// Record successful transfer
    pub async fn record_successful_transfer(&mut self, peer_id: &PeerIdentity) -> Result<()> {
        if let Some(reputation) = self.reputations.get_mut(peer_id) {
            reputation.successful_transfers += 1;
            reputation.trust_score = (reputation.trust_score + 0.1).min(1.0);
            reputation.last_interaction = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
        Ok(())
    }
    
    /// Record failed transfer
    pub async fn record_failed_transfer(&mut self, peer_id: &PeerIdentity) -> Result<()> {
        if let Some(reputation) = self.reputations.get_mut(peer_id) {
            reputation.failed_transfers += 1;
            reputation.trust_score = (reputation.trust_score - 0.2).max(0.0);
            reputation.last_interaction = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
        Ok(())
    }
    
    /// Get reputation statistics
    pub async fn get_stats(&self) -> Result<ReputationStats> {
        let trusted_peers = self.reputations.values()
            .filter(|r| r.trust_score > 0.7)
            .count();
        
        let avg_trust = if !self.reputations.is_empty() {
            self.reputations.values()
                .map(|r| r.trust_score)
                .sum::<f32>() / self.reputations.len() as f32
        } else {
            0.0
        };
        
        Ok(ReputationStats {
            trusted_peer_count: trusted_peers,
            average_trust_score: avg_trust,
        })
    }
}