#!/usr/bin/env python3

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Create a dummy numpy for basic operations
    class DummyNumpy:
        @staticmethod
        def array(*args, **kwargs):
            return list(*args) if args else []
    np = DummyNumpy()
from collections import defaultdict

@dataclass
class ShaderProfile:
    game_id: str
    shader_count: int
    common_features: List[str]
    performance_impact: float
    compile_time_estimate: float

class HeuristicEngine:
    def __init__(self):
        self.game_profiles = {}
        self.global_patterns = defaultdict(float)
        self.steam_deck_optimizations = {
            'max_parallel_compiles': 4,  # Optimal for Steam Deck's 4-core CPU
            'memory_limit_mb': 2048,      # Conservative memory limit
            'priority_boost_popular': 1.5,
            'cache_efficiency_target': 0.85
        }
        
    def analyze_steam_library(self, steam_path: Path) -> Dict:
        """Analyze entire Steam library for shader patterns"""
        library_analysis = {
            'total_games': 0,
            'total_shaders_estimated': 0,
            'common_engines': defaultdict(int),
            'shader_complexity_distribution': {},
            'recommendations': []
        }
        
        # Common game engine shader signatures
        engine_signatures = {
            'Unreal': ['BasePassPixelShader', 'ShadowDepthVertexShader', 'PostProcessCombineLUTs'],
            'Unity': ['Hidden/Internal-', 'Sprites/Default', 'UI/Default'],
            'Source2': ['vr_standard', 'tools_sprite', 'hero_','vr_'],
            'CryEngine': ['IllumDefault', 'ShadowGen', 'PostEffectsGame'],
            'Godot': ['canvas_item', 'spatial', 'particles']
        }
        
        steamapps = steam_path / 'steamapps' / 'common'
        if steamapps.exists():
            for game_dir in steamapps.iterdir():
                if game_dir.is_dir():
                    library_analysis['total_games'] += 1
                    
                    # Quick heuristic scan
                    game_profile = self._profile_game(game_dir, engine_signatures)
                    if game_profile:
                        self.game_profiles[game_dir.name] = game_profile
                        library_analysis['total_shaders_estimated'] += game_profile.shader_count
                        
        # Generate global optimization recommendations
        library_analysis['recommendations'] = self._generate_recommendations()
        return library_analysis
    
    def _profile_game(self, game_path: Path, engine_signatures: Dict) -> ShaderProfile:
        """Create a profile for a specific game"""
        shader_count = 0
        features = []
        engine = 'Unknown'
        
        # Quick scan for engine detection
        for eng_name, signatures in engine_signatures.items():
            for sig in signatures:
                if self._quick_file_search(game_path, sig):
                    engine = eng_name
                    break
                    
        # Estimate shader count based on game size and type
        game_size_mb = self._get_directory_size(game_path) / (1024 * 1024)
        
        # Heuristics based on engine and size
        if engine == 'Unreal':
            shader_count = int(game_size_mb * 0.8)  # Unreal games have many shaders
            features = ['pbr', 'shadow_cascades', 'post_processing', 'temporal_aa']
        elif engine == 'Unity':
            shader_count = int(game_size_mb * 0.5)
            features = ['standard_shader', 'ui_shaders', 'particle_shaders']
        elif engine == 'Source2':
            shader_count = int(game_size_mb * 0.6)
            features = ['pbr', 'volumetric_fog', 'dynamic_lighting']
        else:
            shader_count = int(game_size_mb * 0.3)  # Conservative estimate
            features = ['basic_shading']
            
        performance_impact = min(shader_count / 1000, 5.0)  # 0-5 scale
        compile_time = shader_count * 0.05  # ~50ms per shader estimate
        
        return ShaderProfile(
            game_id=game_path.name,
            shader_count=shader_count,
            common_features=features,
            performance_impact=performance_impact,
            compile_time_estimate=compile_time
        )
    
    def _quick_file_search(self, path: Path, pattern: str) -> bool:
        """Quick search for pattern in directory"""
        try:
            for root, dirs, files in os.walk(path):
                # Limit depth for performance
                if root.count(os.sep) - str(path).count(os.sep) > 3:
                    dirs.clear()
                    continue
                    
                for file in files[:100]:  # Check first 100 files
                    if pattern.lower() in file.lower():
                        return True
                        
        except:
            pass
            
        return False
    
    def _get_directory_size(self, path: Path) -> int:
        """Get directory size in bytes"""
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self._get_directory_size(entry.path)
        except:
            pass
            
        return total
    
    def predict_shader_priorities(self, game_id: str) -> List[Dict]:
        """Predict shader compilation priorities for a specific game"""
        if game_id not in self.game_profiles:
            return []
            
        profile = self.game_profiles[game_id]
        priorities = []
        
        # Base priorities on common shader types
        shader_priority_map = {
            'pbr': {'priority': 10, 'variants': ['forward', 'deferred', 'shadow']},
            'standard_shader': {'priority': 9, 'variants': ['opaque', 'transparent', 'cutout']},
            'shadow_cascades': {'priority': 8, 'variants': ['cascade0', 'cascade1', 'cascade2']},
            'post_processing': {'priority': 7, 'variants': ['bloom', 'tonemap', 'aa']},
            'ui_shaders': {'priority': 6, 'variants': ['default', 'blur', 'mask']},
            'particle_shaders': {'priority': 5, 'variants': ['billboard', 'mesh', 'ribbon']},
            'volumetric_fog': {'priority': 4, 'variants': ['scatter', 'absorb']},
            'temporal_aa': {'priority': 3, 'variants': ['resolve', 'velocity']},
            'basic_shading': {'priority': 2, 'variants': ['diffuse', 'specular']}
        }
        
        for feature in profile.common_features:
            if feature in shader_priority_map:
                shader_info = shader_priority_map[feature]
                for variant in shader_info['variants']:
                    priorities.append({
                        'shader_type': feature,
                        'variant': variant,
                        'priority': shader_info['priority'],
                        'estimated_usage': self._estimate_usage(feature, variant),
                        'compile_flags': self._get_optimal_flags(feature)
                    })
                    
        # Sort by priority and estimated usage
        priorities.sort(key=lambda x: (x['priority'], x['estimated_usage']), reverse=True)
        
        # Apply Steam Deck specific optimizations
        return self._optimize_for_steam_deck(priorities)
    
    def _estimate_usage(self, shader_type: str, variant: str) -> float:
        """Estimate how frequently a shader variant will be used"""
        base_usage = {
            'pbr': 0.9,
            'standard_shader': 0.85,
            'shadow_cascades': 0.8,
            'post_processing': 0.7,
            'ui_shaders': 0.95,
            'particle_shaders': 0.4,
            'volumetric_fog': 0.3,
            'temporal_aa': 0.6,
            'basic_shading': 0.5
        }
        
        variant_multipliers = {
            'forward': 1.0,
            'deferred': 0.8,
            'shadow': 0.9,
            'opaque': 1.0,
            'transparent': 0.6,
            'default': 1.0,
            'bloom': 0.7
        }
        
        base = base_usage.get(shader_type, 0.5)
        multiplier = variant_multipliers.get(variant, 0.8)
        
        return base * multiplier
    
    def _get_optimal_flags(self, shader_type: str) -> List[str]:
        """Get optimal compilation flags for shader type"""
        flags = ['-O2']  # Optimization level 2 by default
        
        if shader_type in ['pbr', 'shadow_cascades']:
            flags.append('-mfp16')  # Use half precision where possible
            
        if shader_type in ['post_processing', 'temporal_aa']:
            flags.append('-munroll-loops')  # Unroll loops for post effects
            
        return flags
    
    def _optimize_for_steam_deck(self, priorities: List[Dict]) -> List[Dict]:
        """Apply Steam Deck specific optimizations"""
        optimized = []
        
        # Limit to what Steam Deck can handle efficiently
        max_concurrent = self.steam_deck_optimizations['max_parallel_compiles']
        memory_per_compile = 50  # MB estimate
        
        current_memory = 0
        for priority in priorities:
            estimated_memory = memory_per_compile
            
            if current_memory + estimated_memory < self.steam_deck_optimizations['memory_limit_mb']:
                # Add Steam Deck specific flags
                priority['compile_flags'].extend(['-mtune=znver2', '-march=znver2'])  # AMD Zen 2
                priority['steam_deck_optimized'] = True
                optimized.append(priority)
                current_memory += estimated_memory
                
        return optimized
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        total_shaders = sum(p.shader_count for p in self.game_profiles.values())
        total_compile_time = sum(p.compile_time_estimate for p in self.game_profiles.values())
        
        recommendations.append(f"Estimated total shaders: {total_shaders:,}")
        recommendations.append(f"Estimated compile time: {total_compile_time/60:.1f} minutes")
        
        # Find most shader-heavy games
        heavy_games = sorted(self.game_profiles.items(), 
                           key=lambda x: x[1].shader_count, 
                           reverse=True)[:5]
        
        if heavy_games:
            recommendations.append("Priority games for shader compilation:")
            for game, profile in heavy_games:
                recommendations.append(f"  - {game}: ~{profile.shader_count} shaders")
                
        return recommendations
    
    def export_fossilize_config(self, output_path: Path) -> None:
        """Export configuration for Fossilize integration"""
        config = {
            'version': '1.0',
            'steam_deck_optimized': True,
            'compilation_hints': {},
            'global_flags': ['-O2', '-mtune=znver2', '-march=znver2'],
            'priority_games': []
        }
        
        # Add per-game hints
        for game_id, profile in self.game_profiles.items():
            priorities = self.predict_shader_priorities(game_id)
            if priorities:
                config['compilation_hints'][game_id] = {
                    'estimated_shaders': profile.shader_count,
                    'priority_shaders': priorities[:20],  # Top 20
                    'features': profile.common_features
                }
                
                if profile.performance_impact > 3.0:
                    config['priority_games'].append(game_id)
                    
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)