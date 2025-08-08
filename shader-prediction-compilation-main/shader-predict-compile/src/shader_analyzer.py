#!/usr/bin/env python3

import os
import re
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import struct

class ShaderAnalyzer:
    def __init__(self):
        self.shader_patterns = defaultdict(int)
        self.shader_hashes = set()
        self.common_includes = defaultdict(int)
        self.shader_stages = defaultdict(int)
        
    def analyze_game_directory(self, game_path: Path) -> Dict:
        """Analyze game files to predict shader patterns"""
        results = {
            'total_shaders': 0,
            'shader_types': defaultdict(int),
            'common_patterns': [],
            'predicted_variants': [],
            'priority_shaders': []
        }
        
        # Common shader extensions and patterns
        shader_extensions = {'.hlsl', '.glsl', '.spv', '.dxbc', '.dxil', '.pso', '.cso'}
        shader_patterns = {
            'vertex': re.compile(r'(vertex|vert|vs)[\._]', re.IGNORECASE),
            'fragment': re.compile(r'(fragment|frag|pixel|ps)[\._]', re.IGNORECASE),
            'compute': re.compile(r'(compute|cs)[\._]', re.IGNORECASE),
            'geometry': re.compile(r'(geometry|geom|gs)[\._]', re.IGNORECASE),
            'tessellation': re.compile(r'(tess|hull|domain|hs|ds)[\._]', re.IGNORECASE)
        }
        
        for root, dirs, files in os.walk(game_path):
            # Skip common non-shader directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules'}]
            
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                if ext in shader_extensions or self._is_shader_file(file_path):
                    results['total_shaders'] += 1
                    
                    # Classify shader type
                    shader_type = self._classify_shader(file_path, shader_patterns)
                    results['shader_types'][shader_type] += 1
                    
                    # Analyze shader content for patterns
                    try:
                        patterns = self._extract_shader_patterns(file_path)
                        for pattern in patterns:
                            self.shader_patterns[pattern] += 1
                    except:
                        pass
                        
        # Generate predictions based on analysis
        results['common_patterns'] = self._get_common_patterns()
        results['predicted_variants'] = self._predict_shader_variants(results['shader_types'])
        results['priority_shaders'] = self._calculate_priorities(results)
        
        return results
    
    def _is_shader_file(self, file_path: Path) -> bool:
        """Heuristic to detect shader files without standard extensions"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(64)
                
            # Check for common shader magic numbers
            if header.startswith(b'DXBC'):  # DirectX bytecode
                return True
            if header.startswith(b'\x03\x02\x23\x07'):  # SPIR-V
                return True
            if b'#version' in header or b'#pragma' in header:  # GLSL
                return True
                
        except:
            pass
            
        return False
    
    def _classify_shader(self, file_path: Path, patterns: Dict) -> str:
        """Classify shader type based on filename and content"""
        filename = file_path.name.lower()
        
        for shader_type, pattern in patterns.items():
            if pattern.search(filename):
                return shader_type
                
        # Try to detect from content
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024).decode('utf-8', errors='ignore')
                
            if 'gl_Position' in content or 'SV_Position' in content:
                return 'vertex'
            elif 'gl_FragColor' in content or 'SV_Target' in content:
                return 'fragment'
            elif 'numthreads' in content.lower():
                return 'compute'
                
        except:
            pass
            
        return 'unknown'
    
    def _extract_shader_patterns(self, file_path: Path) -> List[str]:
        """Extract common shader patterns for prediction"""
        patterns = []
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
                
            # Look for common shader features
            if 'texture' in content.lower():
                patterns.append('texture_sampling')
            if 'shadow' in content.lower():
                patterns.append('shadow_mapping')
            if 'normal' in content.lower():
                patterns.append('normal_mapping')
            if 'skinning' in content.lower() or 'bone' in content.lower():
                patterns.append('skeletal_animation')
            if 'particle' in content.lower():
                patterns.append('particle_system')
            if 'post' in content.lower() or 'screen' in content.lower():
                patterns.append('post_processing')
                
        except:
            pass
            
        return patterns
    
    def _get_common_patterns(self) -> List[Tuple[str, int]]:
        """Get most common shader patterns"""
        return sorted(self.shader_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _predict_shader_variants(self, shader_types: Dict) -> List[Dict]:
        """Predict shader variants based on analysis"""
        variants = []
        
        # Common variant combinations
        if shader_types['vertex'] > 0 and shader_types['fragment'] > 0:
            variants.append({
                'name': 'standard_forward',
                'stages': ['vertex', 'fragment'],
                'features': ['texture_sampling', 'normal_mapping'],
                'priority': 10
            })
            
        if 'shadow_mapping' in self.shader_patterns:
            variants.append({
                'name': 'shadow_pass',
                'stages': ['vertex', 'fragment'],
                'features': ['shadow_mapping'],
                'priority': 8
            })
            
        if 'skeletal_animation' in self.shader_patterns:
            variants.append({
                'name': 'animated_mesh',
                'stages': ['vertex', 'fragment'],
                'features': ['skeletal_animation', 'texture_sampling'],
                'priority': 7
            })
            
        if shader_types['compute'] > 0:
            variants.append({
                'name': 'compute_dispatch',
                'stages': ['compute'],
                'features': [],
                'priority': 6
            })
            
        return variants
    
    def _calculate_priorities(self, results: Dict) -> List[Dict]:
        """Calculate shader compilation priorities"""
        priorities = []
        
        # Prioritize based on usage frequency and complexity
        for variant in results['predicted_variants']:
            priority_score = variant['priority']
            
            # Adjust based on pattern frequency
            for feature in variant['features']:
                if feature in self.shader_patterns:
                    priority_score += self.shader_patterns[feature] * 0.1
                    
            priorities.append({
                'variant': variant['name'],
                'score': priority_score,
                'estimated_compile_time': len(variant['stages']) * 0.5
            })
            
        return sorted(priorities, key=lambda x: x['score'], reverse=True)
    
    def generate_fossilize_hints(self, analysis_results: Dict) -> Dict:
        """Generate hints for Fossilize shader compilation"""
        hints = {
            'version': '1.0',
            'priority_shaders': [],
            'shader_variants': [],
            'compilation_order': []
        }
        
        # Convert analysis to Fossilize-compatible format
        for priority in analysis_results['priority_shaders']:
            hints['priority_shaders'].append({
                'variant_name': priority['variant'],
                'priority_score': priority['score'],
                'stages': analysis_results['predicted_variants'][0]['stages']
            })
            
        # Generate compilation order
        hints['compilation_order'] = [p['variant'] for p in analysis_results['priority_shaders']]
        
        return hints