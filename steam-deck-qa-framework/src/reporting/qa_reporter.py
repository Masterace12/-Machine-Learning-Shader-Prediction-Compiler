#!/usr/bin/env python3
"""
QA Reporting Module
Generates comprehensive reports for testing results
"""

import os
import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from datetime import datetime
from jinja2 import Template
import numpy as np

class QAReporter:
    """Comprehensive QA reporting and visualization"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}")
        self.report_directory = "data/reports"
        self.template_directory = "config/templates"
        self._setup_reporting_environment()
    
    def _setup_reporting_environment(self):
        """Setup reporting environment and directories"""
        try:
            os.makedirs(self.report_directory, exist_ok=True)
            os.makedirs(self.template_directory, exist_ok=True)
            
            # Set matplotlib style for consistent plots
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
        except Exception as e:
            self.logger.error(f"Failed to setup reporting environment: {e}")
    
    async def generate_comprehensive_report(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Generate comprehensive HTML report"""
        self.logger.info(f"Generating comprehensive report for session {session_id}")
        
        try:
            # Generate visualizations
            charts = await self._generate_charts(test_results, session_id)
            
            # Prepare report data
            report_data = {
                "session_id": session_id,
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_results": test_results,
                "charts": charts,
                "summary": test_results.get("summary", {}),
                "system_info": test_results.get("system_info", {}),
                "recommendations": self._generate_comprehensive_recommendations(test_results)
            }
            
            # Generate HTML report
            html_content = await self._generate_html_report(report_data)
            
            # Save report
            report_path = f"{self.report_directory}/comprehensive_report_{session_id}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Comprehensive report saved to {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return ""
    
    async def generate_executive_summary(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Generate executive summary report"""
        self.logger.info(f"Generating executive summary for session {session_id}")
        
        try:
            summary_data = {
                "session_id": session_id,
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_status": self._determine_overall_status(test_results),
                "key_metrics": self._extract_key_metrics(test_results),
                "critical_issues": test_results.get("summary", {}).get("critical_issues", []),
                "pass_rate": test_results.get("summary", {}).get("pass_rate", 0.0),
                "total_games": test_results.get("summary", {}).get("total_games_tested", 0),
                "recommendations": self._generate_executive_recommendations(test_results)
            }
            
            # Generate executive summary
            summary_content = await self._generate_executive_summary_content(summary_data)
            
            # Save summary
            summary_path = f"{self.report_directory}/executive_summary_{session_id}.html"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            
            self.logger.info(f"Executive summary saved to {summary_path}")
            return summary_path
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return ""
    
    async def generate_compatibility_matrix(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Generate compatibility matrix visualization"""
        self.logger.info(f"Generating compatibility matrix for session {session_id}")
        
        try:
            # Extract compatibility data
            compatibility_data = self._extract_compatibility_data(test_results)
            
            # Create compatibility matrix
            matrix_path = await self._create_compatibility_matrix(compatibility_data, session_id)
            
            self.logger.info(f"Compatibility matrix saved to {matrix_path}")
            return matrix_path
            
        except Exception as e:
            self.logger.error(f"Error generating compatibility matrix: {e}")
            return ""
    
    async def generate_regression_report(self, baseline_results: Dict[str, Any], 
                                       current_results: Dict[str, Any], 
                                       regression_analysis: Dict[str, Any], 
                                       session_id: str) -> str:
        """Generate regression analysis report"""
        self.logger.info(f"Generating regression report for session {session_id}")
        
        try:
            # Create regression visualizations
            regression_charts = await self._generate_regression_charts(
                baseline_results, current_results, regression_analysis, session_id
            )
            
            # Prepare regression report data
            report_data = {
                "session_id": session_id,
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "regression_analysis": regression_analysis,
                "baseline_summary": baseline_results.get("summary", {}),
                "current_summary": current_results.get("summary", {}),
                "charts": regression_charts,
                "recommendations": self._generate_regression_recommendations(regression_analysis)
            }
            
            # Generate HTML report
            html_content = await self._generate_regression_report_html(report_data)
            
            # Save report
            report_path = f"{self.report_directory}/regression_report_{session_id}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Regression report saved to {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating regression report: {e}")
            return ""
    
    async def _generate_charts(self, test_results: Dict[str, Any], session_id: str) -> Dict[str, str]:
        """Generate all charts for the comprehensive report"""
        charts = {}
        
        try:
            # 1. Overall performance chart
            perf_chart = await self._create_performance_chart(test_results, session_id)
            charts["performance"] = perf_chart
            
            # 2. Compatibility status chart
            compat_chart = await self._create_compatibility_chart(test_results, session_id)
            charts["compatibility"] = compat_chart
            
            # 3. Anti-cheat validation chart
            anticheat_chart = await self._create_anticheat_chart(test_results, session_id)
            charts["anticheat"] = anticheat_chart
            
            # 4. Shader cache metrics chart
            cache_chart = await self._create_cache_metrics_chart(test_results, session_id)
            charts["cache_metrics"] = cache_chart
            
            # 5. Performance trends chart
            trends_chart = await self._create_performance_trends_chart(test_results, session_id)
            charts["performance_trends"] = trends_chart
            
            # 6. ML prediction accuracy chart
            ml_chart = await self._create_ml_accuracy_chart(test_results, session_id)
            charts["ml_accuracy"] = ml_chart
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
        
        return charts
    
    async def _create_performance_chart(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Create performance overview chart"""
        try:
            game_results = test_results.get("game_results", {})
            
            games = []
            fps_values = []
            stutter_counts = []
            
            for game, result in game_results.items():
                perf_analysis = result.get("performance_analysis", {})
                overall_perf = perf_analysis.get("overall_performance", {})
                stutter_analysis = perf_analysis.get("stutter_analysis", {})
                
                games.append(game.replace('_', ' ').title())
                fps_values.append(overall_perf.get("average_fps_all_scenarios", 0))
                stutter_counts.append(stutter_analysis.get("total_stutter_events", 0))
            
            if not games:
                return ""
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # FPS chart
            bars1 = ax1.bar(games, fps_values, color='skyblue', alpha=0.7)
            ax1.axhline(y=60, color='red', linestyle='--', label='Target FPS (60)')
            ax1.set_ylabel('Average FPS')
            ax1.set_title('Game Performance - Average FPS')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, fps_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value:.1f}', ha='center', va='bottom')
            
            # Stutter events chart
            bars2 = ax2.bar(games, stutter_counts, color='lightcoral', alpha=0.7)
            ax2.set_ylabel('Stutter Events')
            ax2.set_title('Game Performance - Stutter Events')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars2, stutter_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{int(value)}', ha='center', va='bottom')
            
            # Rotate x-axis labels
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/performance_chart_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating performance chart: {e}")
            return ""
    
    async def _create_compatibility_chart(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Create compatibility status pie chart"""
        try:
            game_results = test_results.get("game_results", {})
            
            status_counts = {"passed": 0, "failed": 0, "warning": 0}
            
            for result in game_results.values():
                status = result.get("status", "failed")
                if status in status_counts:
                    status_counts[status] += 1
            
            if sum(status_counts.values()) == 0:
                return ""
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            labels = []
            sizes = []
            colors = []
            
            color_map = {
                "passed": "#4CAF50",    # Green
                "warning": "#FF9800",   # Orange
                "failed": "#F44336"     # Red
            }
            
            for status, count in status_counts.items():
                if count > 0:
                    labels.append(f"{status.title()} ({count})")
                    sizes.append(count)
                    colors.append(color_map[status])
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Game Compatibility Status Distribution', fontsize=16, fontweight='bold')
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/compatibility_chart_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating compatibility chart: {e}")
            return ""
    
    async def _create_anticheat_chart(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Create anti-cheat compatibility chart"""
        try:
            game_results = test_results.get("game_results", {})
            
            anticheat_data = {}
            
            for game, result in game_results.items():
                anticheat_validation = result.get("anticheat_validation", {})
                if anticheat_validation:  # Only if anti-cheat was tested
                    compatible = anticheat_validation.get("compatible", False)
                    anticheat_type = anticheat_validation.get("anticheat_type", "unknown")
                    
                    if anticheat_type not in anticheat_data:
                        anticheat_data[anticheat_type] = {"compatible": 0, "incompatible": 0}
                    
                    if compatible:
                        anticheat_data[anticheat_type]["compatible"] += 1
                    else:
                        anticheat_data[anticheat_type]["incompatible"] += 1
            
            if not anticheat_data:
                return ""
            
            # Create stacked bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            anticheat_systems = list(anticheat_data.keys())
            compatible_counts = [anticheat_data[ac]["compatible"] for ac in anticheat_systems]
            incompatible_counts = [anticheat_data[ac]["incompatible"] for ac in anticheat_systems]
            
            x = np.arange(len(anticheat_systems))
            width = 0.35
            
            bars1 = ax.bar(x, compatible_counts, width, label='Compatible', color='#4CAF50')
            bars2 = ax.bar(x, incompatible_counts, width, bottom=compatible_counts, 
                          label='Incompatible', color='#F44336')
            
            ax.set_xlabel('Anti-Cheat System')
            ax.set_ylabel('Number of Games')
            ax.set_title('Anti-Cheat System Compatibility')
            ax.set_xticks(x)
            ax.set_xticklabels([ac.upper() for ac in anticheat_systems])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_y() + height/2,
                               f'{int(height)}', ha='center', va='center',
                               color='white', fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/anticheat_chart_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating anti-cheat chart: {e}")
            return ""
    
    async def _create_cache_metrics_chart(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Create shader cache metrics chart"""
        try:
            game_results = test_results.get("game_results", {})
            
            games = []
            cache_sizes = []
            hit_rates = []
            corruption_rates = []
            
            for game, result in game_results.items():
                cache_validation = result.get("cache_validation", {})
                if cache_validation:
                    cache_files = cache_validation.get("cache_files", {})
                    performance_metrics = cache_validation.get("performance_metrics", {})
                    
                    games.append(game.replace('_', ' ').title())
                    cache_sizes.append(cache_files.get("total_size", 0) / (1024 * 1024))  # Convert to MB
                    hit_rates.append(performance_metrics.get("hit_ratio_estimate", 0) * 100)
                    corruption_rates.append(cache_files.get("corruption_rate", 0) * 100)
            
            if not games:
                return ""
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Cache sizes
            bars1 = ax1.bar(games, cache_sizes, color='lightblue', alpha=0.7)
            ax1.set_ylabel('Cache Size (MB)')
            ax1.set_title('Shader Cache Sizes')
            ax1.grid(True, alpha=0.3)
            
            for bar, value in zip(bars1, cache_sizes):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cache_sizes)*0.01, 
                        f'{value:.1f}', ha='center', va='bottom', fontsize=8)
            
            # Hit rates
            bars2 = ax2.bar(games, hit_rates, color='lightgreen', alpha=0.7)
            ax2.axhline(y=85, color='red', linestyle='--', label='Target (85%)')
            ax2.set_ylabel('Hit Rate (%)')
            ax2.set_title('Cache Hit Rates')
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            for bar, value in zip(bars2, hit_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Corruption rates
            bars3 = ax3.bar(games, corruption_rates, color='lightcoral', alpha=0.7)
            ax3.axhline(y=5, color='red', linestyle='--', label='Acceptable Limit (5%)')
            ax3.set_ylabel('Corruption Rate (%)')
            ax3.set_title('Cache Corruption Rates')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars3, corruption_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # Cache efficiency scatter plot
            if len(cache_sizes) == len(hit_rates):
                scatter = ax4.scatter(cache_sizes, hit_rates, 
                                    s=[100]*len(games), alpha=0.6, c=range(len(games)))
                ax4.set_xlabel('Cache Size (MB)')
                ax4.set_ylabel('Hit Rate (%)')
                ax4.set_title('Cache Size vs Hit Rate')
                ax4.grid(True, alpha=0.3)
                
                # Add game labels
                for i, game in enumerate(games):
                    ax4.annotate(game, (cache_sizes[i], hit_rates[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Rotate x-axis labels
            for ax in [ax1, ax2, ax3]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/cache_metrics_chart_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating cache metrics chart: {e}")
            return ""
    
    async def _create_performance_trends_chart(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Create performance trends over time chart"""
        try:
            # This would require historical data
            # For now, create a placeholder chart showing current session data
            
            game_results = test_results.get("game_results", {})
            
            # Extract performance data over test scenarios (simulating time series)
            scenario_data = {}
            
            for game, result in game_results.items():
                perf_analysis = result.get("performance_analysis", {})
                test_scenarios = perf_analysis.get("test_scenarios", {})
                
                for scenario, scenario_result in test_scenarios.items():
                    perf_metrics = scenario_result.get("performance_metrics", {})
                    avg_fps = perf_metrics.get("average_fps", 0)
                    
                    if scenario not in scenario_data:
                        scenario_data[scenario] = []
                    scenario_data[scenario].append(avg_fps)
            
            if not scenario_data:
                return ""
            
            # Create line chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for scenario, fps_values in scenario_data.items():
                if fps_values:
                    x_values = range(len(fps_values))
                    ax.plot(x_values, fps_values, marker='o', label=scenario.replace('_', ' ').title())
            
            ax.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='Target FPS (60)')
            ax.set_xlabel('Game Index')
            ax.set_ylabel('Average FPS')
            ax.set_title('Performance Trends Across Games by Scenario')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/performance_trends_chart_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating performance trends chart: {e}")
            return ""
    
    async def _create_ml_accuracy_chart(self, test_results: Dict[str, Any], session_id: str) -> str:
        """Create ML prediction accuracy chart"""
        try:
            game_results = test_results.get("game_results", {})
            
            games = []
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            
            for game, result in game_results.items():
                ml_metrics = result.get("ml_metrics", {})
                aggregated = ml_metrics.get("aggregated_metrics")
                
                if aggregated:
                    games.append(game.replace('_', ' ').title())
                    
                    # Calculate accuracy from confusion matrix
                    tp = getattr(aggregated, 'true_positives', 0)
                    tn = getattr(aggregated, 'true_negatives', 0)
                    fp = getattr(aggregated, 'false_positives', 0)
                    fn = getattr(aggregated, 'false_negatives', 0)
                    
                    total = tp + tn + fp + fn
                    accuracy = (tp + tn) / total if total > 0 else 0
                    
                    accuracy_scores.append(accuracy * 100)
                    precision_scores.append(getattr(aggregated, 'precision', 0) * 100)
                    recall_scores.append(getattr(aggregated, 'recall', 0) * 100)
            
            if not games:
                return ""
            
            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(games))
            width = 0.25
            
            bars1 = ax.bar(x - width, accuracy_scores, width, label='Accuracy', color='skyblue')
            bars2 = ax.bar(x, precision_scores, width, label='Precision', color='lightgreen')
            bars3 = ax.bar(x + width, recall_scores, width, label='Recall', color='lightcoral')
            
            ax.set_xlabel('Games')
            ax.set_ylabel('Score (%)')
            ax.set_title('ML Prediction Model Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(games, rotation=45)
            ax.legend()
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/ml_accuracy_chart_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating ML accuracy chart: {e}")
            return ""
    
    async def _create_compatibility_matrix(self, compatibility_data: Dict[str, Any], session_id: str) -> str:
        """Create compatibility matrix heatmap"""
        try:
            # Create compatibility matrix
            games = list(compatibility_data.keys())
            test_categories = ["Compatibility", "Anti-Cheat", "Cache", "Performance", "ML Prediction"]
            
            # Create matrix data
            matrix_data = []
            for game in games:
                row = []
                game_data = compatibility_data[game]
                
                # Compatibility (0=failed, 1=warning, 2=passed)
                status = game_data.get("status", "failed")
                status_val = {"failed": 0, "warning": 1, "passed": 2}[status]
                row.append(status_val)
                
                # Anti-cheat (0=incompatible, 1=unknown, 2=compatible)
                anticheat = game_data.get("anticheat_compatible", None)
                if anticheat is None:
                    row.append(1)  # Unknown
                elif anticheat:
                    row.append(2)  # Compatible
                else:
                    row.append(0)  # Incompatible
                
                # Cache (0=invalid, 1=warning, 2=valid)
                cache_valid = game_data.get("cache_valid", False)
                row.append(2 if cache_valid else 0)
                
                # Performance (based on FPS achievement)
                fps_achievement = game_data.get("fps_achievement", 0)
                if fps_achievement >= 90:
                    row.append(2)
                elif fps_achievement >= 70:
                    row.append(1)
                else:
                    row.append(0)
                
                # ML Prediction (based on F1 score)
                f1_score = game_data.get("ml_f1_score", 0)
                if f1_score >= 0.8:
                    row.append(2)
                elif f1_score >= 0.6:
                    row.append(1)
                else:
                    row.append(0)
                
                matrix_data.append(row)
            
            if not matrix_data:
                return ""
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, max(8, len(games) * 0.5)))
            
            # Custom colormap
            colors = ['#F44336', '#FF9800', '#4CAF50']  # Red, Orange, Green
            n_bins = 3
            cmap = plt.matplotlib.colors.ListedColormap(colors)
            
            im = ax.imshow(matrix_data, cmap=cmap, aspect='auto', vmin=0, vmax=2)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(test_categories)))
            ax.set_yticks(np.arange(len(games)))
            ax.set_xticklabels(test_categories)
            ax.set_yticklabels([game.replace('_', ' ').title() for game in games])
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            for i in range(len(games)):
                for j in range(len(test_categories)):
                    value = matrix_data[i][j]
                    status_text = {0: "FAIL", 1: "WARN", 2: "PASS"}[value]
                    text_color = "white" if value in [0, 2] else "black"
                    ax.text(j, i, status_text, ha="center", va="center", 
                           color=text_color, fontweight="bold")
            
            ax.set_title("Game Compatibility Matrix")
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6)
            cbar.set_ticks([0, 1, 2])
            cbar.set_ticklabels(['Failed', 'Warning', 'Passed'])
            
            plt.tight_layout()
            
            # Save matrix
            matrix_path = f"{self.report_directory}/compatibility_matrix_{session_id}.png"
            plt.savefig(matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return matrix_path
            
        except Exception as e:
            self.logger.error(f"Error creating compatibility matrix: {e}")
            return ""
    
    def _extract_compatibility_data(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract compatibility data for matrix visualization"""
        compatibility_data = {}
        
        try:
            game_results = test_results.get("game_results", {})
            
            for game, result in game_results.items():
                game_compat = {
                    "status": result.get("status", "failed"),
                    "anticheat_compatible": None,
                    "cache_valid": False,
                    "fps_achievement": 0,
                    "ml_f1_score": 0
                }
                
                # Anti-cheat compatibility
                anticheat_validation = result.get("anticheat_validation", {})
                if anticheat_validation:
                    game_compat["anticheat_compatible"] = anticheat_validation.get("compatible", False)
                
                # Cache validity
                cache_validation = result.get("cache_validation", {})
                game_compat["cache_valid"] = cache_validation.get("valid", False)
                
                # Performance achievement
                perf_analysis = result.get("performance_analysis", {})
                overall_perf = perf_analysis.get("overall_performance", {})
                target_achievement = overall_perf.get("target_fps_achievement", 0)
                game_compat["fps_achievement"] = target_achievement
                
                # ML F1 score
                ml_metrics = result.get("ml_metrics", {})
                aggregated = ml_metrics.get("aggregated_metrics")
                if aggregated:
                    game_compat["ml_f1_score"] = getattr(aggregated, 'f1_score', 0)
                
                compatibility_data[game] = game_compat
        
        except Exception as e:
            self.logger.error(f"Error extracting compatibility data: {e}")
        
        return compatibility_data
    
    async def _generate_regression_charts(self, baseline_results: Dict[str, Any], 
                                        current_results: Dict[str, Any], 
                                        regression_analysis: Dict[str, Any], 
                                        session_id: str) -> Dict[str, str]:
        """Generate regression analysis charts"""
        charts = {}
        
        try:
            # Performance comparison chart
            perf_chart = await self._create_performance_comparison_chart(
                baseline_results, current_results, session_id
            )
            charts["performance_comparison"] = perf_chart
            
            # Regression impact chart
            regression_chart = await self._create_regression_impact_chart(
                regression_analysis, session_id
            )
            charts["regression_impact"] = regression_chart
            
        except Exception as e:
            self.logger.error(f"Error generating regression charts: {e}")
        
        return charts
    
    async def _create_performance_comparison_chart(self, baseline_results: Dict[str, Any], 
                                                 current_results: Dict[str, Any], 
                                                 session_id: str) -> str:
        """Create performance comparison chart between baseline and current"""
        try:
            baseline_games = baseline_results.get("game_results", {})
            current_games = current_results.get("game_results", {})
            
            games = []
            baseline_fps = []
            current_fps = []
            
            for game in baseline_games:
                if game in current_games:
                    baseline_perf = baseline_games[game].get("performance_analysis", {}).get("overall_performance", {})
                    current_perf = current_games[game].get("performance_analysis", {}).get("overall_performance", {})
                    
                    baseline_avg = baseline_perf.get("average_fps_all_scenarios", 0)
                    current_avg = current_perf.get("average_fps_all_scenarios", 0)
                    
                    if baseline_avg > 0 and current_avg > 0:
                        games.append(game.replace('_', ' ').title())
                        baseline_fps.append(baseline_avg)
                        current_fps.append(current_avg)
            
            if not games:
                return ""
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(games))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, baseline_fps, width, label='Baseline', color='lightblue')
            bars2 = ax.bar(x + width/2, current_fps, width, label='Current', color='lightcoral')
            
            ax.set_xlabel('Games')
            ax.set_ylabel('Average FPS')
            ax.set_title('Performance Comparison: Baseline vs Current')
            ax.set_xticks(x)
            ax.set_xticklabels(games, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels and change indicators
            for i, (baseline, current) in enumerate(zip(baseline_fps, current_fps)):
                # Baseline label
                ax.text(bars1[i].get_x() + bars1[i].get_width()/2, baseline + 1,
                       f'{baseline:.1f}', ha='center', va='bottom', fontsize=8)
                
                # Current label
                ax.text(bars2[i].get_x() + bars2[i].get_width()/2, current + 1,
                       f'{current:.1f}', ha='center', va='bottom', fontsize=8)
                
                # Change indicator
                change = current - baseline
                change_pct = (change / baseline) * 100 if baseline > 0 else 0
                color = 'green' if change > 0 else 'red' if change < 0 else 'gray'
                
                ax.annotate(f'{change_pct:+.1f}%', 
                           xy=(i, max(baseline, current) + 3),
                           ha='center', va='bottom', color=color, fontweight='bold')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/performance_comparison_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating performance comparison chart: {e}")
            return ""
    
    async def _create_regression_impact_chart(self, regression_analysis: Dict[str, Any], session_id: str) -> str:
        """Create regression impact visualization"""
        try:
            # Extract regression data
            new_failures = regression_analysis.get("new_failures", [])
            fixed_issues = regression_analysis.get("fixed_issues", [])
            performance_regressions = regression_analysis.get("performance_regressions", [])
            
            # Create summary chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Regression summary pie chart
            categories = []
            counts = []
            colors = []
            
            if new_failures:
                categories.append(f'New Failures ({len(new_failures)})')
                counts.append(len(new_failures))
                colors.append('#F44336')
            
            if performance_regressions:
                categories.append(f'Performance Regressions ({len(performance_regressions)})')
                counts.append(len(performance_regressions))
                colors.append('#FF9800')
            
            if fixed_issues:
                categories.append(f'Fixed Issues ({len(fixed_issues)})')
                counts.append(len(fixed_issues))
                colors.append('#4CAF50')
            
            if counts:
                wedges, texts, autotexts = ax1.pie(counts, labels=categories, colors=colors, 
                                                  autopct='%1.1f%%', startangle=90)
                ax1.set_title('Regression Analysis Summary')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                ax1.text(0.5, 0.5, 'No Regressions Detected', ha='center', va='center', 
                        fontsize=16, transform=ax1.transAxes)
                ax1.set_title('Regression Analysis Summary')
            
            # Performance regression details
            if performance_regressions:
                games = [reg["game"] for reg in performance_regressions]
                regression_pcts = []
                
                for reg in performance_regressions:
                    details = reg.get("details", {})
                    if "regression_percent" in details:
                        regression_pcts.append(details["regression_percent"])
                    else:
                        regression_pcts.append(0)
                
                bars = ax2.bar(games, regression_pcts, color='#FF5722', alpha=0.7)
                ax2.set_ylabel('Performance Regression (%)')
                ax2.set_title('Performance Regression by Game')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, regression_pcts):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}%', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'No Performance Regressions', ha='center', va='center', 
                        fontsize=14, transform=ax2.transAxes)
                ax2.set_title('Performance Regression by Game')
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f"{self.report_directory}/regression_impact_{session_id}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return os.path.basename(chart_path)
            
        except Exception as e:
            self.logger.error(f"Error creating regression impact chart: {e}")
            return ""
    
    def _determine_overall_status(self, test_results: Dict[str, Any]) -> str:
        """Determine overall status for executive summary"""
        try:
            summary = test_results.get("summary", {})
            critical_issues = summary.get("critical_issues", [])
            pass_rate = summary.get("pass_rate", 0.0)
            
            if critical_issues:
                return "CRITICAL"
            elif pass_rate >= 0.9:
                return "EXCELLENT"
            elif pass_rate >= 0.8:
                return "GOOD"
            elif pass_rate >= 0.6:
                return "ACCEPTABLE"
            else:
                return "POOR"
        
        except Exception:
            return "UNKNOWN"
    
    def _extract_key_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for executive summary"""
        key_metrics = {
            "total_games": 0,
            "passed_games": 0,
            "failed_games": 0,
            "average_fps": 0.0,
            "severe_stutters": 0,
            "cache_issues": 0,
            "anticheat_issues": 0
        }
        
        try:
            summary = test_results.get("summary", {})
            game_results = test_results.get("game_results", {})
            
            key_metrics["total_games"] = summary.get("total_games_tested", 0)
            key_metrics["passed_games"] = summary.get("passed", 0)
            key_metrics["failed_games"] = summary.get("failed", 0)
            
            # Calculate average FPS across all games
            fps_values = []
            severe_stutters = 0
            cache_issues = 0
            anticheat_issues = 0
            
            for result in game_results.values():
                # FPS
                perf_analysis = result.get("performance_analysis", {})
                overall_perf = perf_analysis.get("overall_performance", {})
                avg_fps = overall_perf.get("average_fps_all_scenarios", 0)
                if avg_fps > 0:
                    fps_values.append(avg_fps)
                
                # Severe stutters
                stutter_analysis = perf_analysis.get("stutter_analysis", {})
                severe_count = stutter_analysis.get("stutter_severity_breakdown", {}).get("severe", 0)
                severe_stutters += severe_count
                
                # Cache issues
                cache_validation = result.get("cache_validation", {})
                if not cache_validation.get("valid", True):
                    cache_issues += 1
                
                # Anti-cheat issues
                anticheat_validation = result.get("anticheat_validation", {})
                if anticheat_validation and not anticheat_validation.get("compatible", True):
                    anticheat_issues += 1
            
            key_metrics["average_fps"] = sum(fps_values) / len(fps_values) if fps_values else 0
            key_metrics["severe_stutters"] = severe_stutters
            key_metrics["cache_issues"] = cache_issues
            key_metrics["anticheat_issues"] = anticheat_issues
        
        except Exception as e:
            self.logger.error(f"Error extracting key metrics: {e}")
        
        return key_metrics
    
    def _generate_comprehensive_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        try:
            summary = test_results.get("summary", {})
            game_results = test_results.get("game_results", {})
            
            # Critical issues
            critical_issues = summary.get("critical_issues", [])
            if critical_issues:
                recommendations.append("CRITICAL: Address the following critical issues immediately:")
                for issue in critical_issues[:5]:  # Limit to top 5
                    recommendations.append(f"  • {issue}")
            
            # Performance recommendations
            low_fps_games = []
            high_stutter_games = []
            
            for game, result in game_results.items():
                perf_analysis = result.get("performance_analysis", {})
                overall_perf = perf_analysis.get("overall_performance", {})
                avg_fps = overall_perf.get("average_fps_all_scenarios", 0)
                
                if avg_fps < 45:  # Below acceptable threshold
                    low_fps_games.append(game)
                
                stutter_analysis = perf_analysis.get("stutter_analysis", {})
                severe_stutters = stutter_analysis.get("stutter_severity_breakdown", {}).get("severe", 0)
                if severe_stutters > 10:
                    high_stutter_games.append(game)
            
            if low_fps_games:
                recommendations.append(f"Performance optimization needed for: {', '.join(low_fps_games[:3])}")
            
            if high_stutter_games:
                recommendations.append(f"Address severe stuttering in: {', '.join(high_stutter_games[:3])}")
            
            # Cache recommendations
            cache_issues = sum(1 for result in game_results.values() 
                             if not result.get("cache_validation", {}).get("valid", True))
            if cache_issues > 0:
                recommendations.append(f"Shader cache issues detected in {cache_issues} games - investigate corruption")
            
            # Anti-cheat recommendations
            anticheat_issues = sum(1 for result in game_results.values() 
                                 if result.get("anticheat_validation", {}) and 
                                 not result.get("anticheat_validation", {}).get("compatible", True))
            if anticheat_issues > 0:
                recommendations.append(f"Anti-cheat compatibility issues in {anticheat_issues} games")
            
            # ML model recommendations
            low_ml_accuracy = sum(1 for result in game_results.values() 
                                if hasattr(result.get("ml_metrics", {}).get("aggregated_metrics"), 'f1_score') and
                                result.get("ml_metrics", {}).get("aggregated_metrics").f1_score < 0.7)
            
            if low_ml_accuracy > 0:
                recommendations.append(f"ML prediction model needs improvement for {low_ml_accuracy} games")
            
            if not recommendations:
                recommendations.append("All systems are performing within acceptable parameters")
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations
    
    def _generate_executive_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate executive-level recommendations"""
        recommendations = []
        
        try:
            summary = test_results.get("summary", {})
            pass_rate = summary.get("pass_rate", 0.0)
            critical_issues = summary.get("critical_issues", [])
            
            # High-level status assessment
            if critical_issues:
                recommendations.append("IMMEDIATE ACTION REQUIRED: Critical compatibility issues detected")
                recommendations.append("Deploy hotfix before general availability")
            
            elif pass_rate < 0.6:
                recommendations.append("HOLD RELEASE: Pass rate below acceptable threshold")
                recommendations.append("Comprehensive testing and optimization required")
            
            elif pass_rate < 0.8:
                recommendations.append("CAUTION: Several games need optimization")
                recommendations.append("Consider targeted fixes before release")
            
            else:
                recommendations.append("READY FOR RELEASE: All systems performing well")
                recommendations.append("Continue monitoring for edge cases")
            
            # Resource allocation recommendations
            failed_count = summary.get("failed", 0)
            if failed_count > 0:
                recommendations.append(f"Allocate development resources to fix {failed_count} failed games")
            
            # Strategic recommendations
            if len(critical_issues) > 3:
                recommendations.append("Consider extending QA cycle to address systemic issues")
            
        except Exception as e:
            self.logger.error(f"Error generating executive recommendations: {e}")
            recommendations.append("Contact QA team for detailed analysis")
        
        return recommendations
    
    def _generate_regression_recommendations(self, regression_analysis: Dict[str, Any]) -> List[str]:
        """Generate regression-specific recommendations"""
        recommendations = []
        
        try:
            new_failures = regression_analysis.get("new_failures", [])
            performance_regressions = regression_analysis.get("performance_regressions", [])
            overall_regression = regression_analysis.get("overall_regression", False)
            
            if overall_regression:
                recommendations.append("REGRESSION DETECTED: Do not proceed with release")
                
                if new_failures:
                    recommendations.append(f"Fix new failures in: {', '.join(new_failures)}")
                
                if performance_regressions:
                    perf_games = [reg["game"] for reg in performance_regressions]
                    recommendations.append(f"Address performance regressions in: {', '.join(perf_games)}")
            
            else:
                fixed_issues = regression_analysis.get("fixed_issues", [])
                if fixed_issues:
                    recommendations.append(f"Good: Fixed issues in {', '.join(fixed_issues)}")
                
                recommendations.append("No regressions detected - safe to proceed")
        
        except Exception as e:
            self.logger.error(f"Error generating regression recommendations: {e}")
            recommendations.append("Manual review required for regression analysis")
        
        return recommendations
    
    async def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        try:
            # Create HTML template
            html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steam Deck QA Framework - Comprehensive Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 3px solid #2196F3; padding-bottom: 20px; }
        .summary { display: flex; justify-content: space-around; margin-bottom: 30px; }
        .metric { text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #2196F3; }
        .metric-label { font-size: 0.9em; color: #666; margin-top: 5px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .game-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .game-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #fafafa; }
        .status-passed { border-left: 5px solid #4CAF50; }
        .status-warning { border-left: 5px solid #FF9800; }
        .status-failed { border-left: 5px solid #F44336; }
        .recommendations { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 15px; }
        .recommendations ul { margin: 10px 0; padding-left: 20px; }
        .critical { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Steam Deck QA Framework</h1>
            <h2>Comprehensive Test Report</h2>
            <p>Session ID: {{ session_id }} | Generated: {{ generation_time }}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{{ summary.total_games_tested or 0 }}</div>
                <div class="metric-label">Games Tested</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(summary.pass_rate * 100) }}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ summary.passed or 0 }}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ summary.failed or 0 }}</div>
                <div class="metric-label">Failed</div>
            </div>
        </div>

        {% if summary.critical_issues %}
        <div class="section">
            <h2>Critical Issues</h2>
            <div class="recommendations critical">
                <h3>⚠️ Immediate Attention Required</h3>
                <ul>
                {% for issue in summary.critical_issues %}
                    <li>{{ issue }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>Performance Overview</h2>
            <div class="chart">
                {% if charts.performance %}
                <img src="{{ charts.performance }}" alt="Performance Chart">
                {% endif %}
            </div>
        </div>

        <div class="section">
            <h2>Compatibility Status</h2>
            <div class="chart">
                {% if charts.compatibility %}
                <img src="{{ charts.compatibility }}" alt="Compatibility Chart">
                {% endif %}
            </div>
        </div>

        <div class="section">
            <h2>Game Results</h2>
            <div class="game-grid">
                {% for game_name, result in test_results.game_results.items() %}
                <div class="game-card status-{{ result.status }}">
                    <h3>{{ game_name.replace('_', ' ').title() }}</h3>
                    <p><strong>Status:</strong> {{ result.status.title() }}</p>
                    
                    {% if result.compatibility_test %}
                    <p><strong>Compatibility:</strong> {{ result.compatibility_test.status }}</p>
                    {% endif %}
                    
                    {% if result.performance_analysis %}
                    <p><strong>Performance:</strong> 
                    {{ "%.1f"|format(result.performance_analysis.overall_performance.average_fps_all_scenarios or 0) }} FPS avg</p>
                    {% endif %}
                    
                    {% if result.cache_validation %}
                    <p><strong>Cache:</strong> 
                    {% if result.cache_validation.valid %}✅ Valid{% else %}❌ Invalid{% endif %}</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </div>

        {% if charts.cache_metrics %}
        <div class="section">
            <h2>Shader Cache Metrics</h2>
            <div class="chart">
                <img src="{{ charts.cache_metrics }}" alt="Cache Metrics Chart">
            </div>
        </div>
        {% endif %}

        {% if charts.anticheat %}
        <div class="section">
            <h2>Anti-Cheat Compatibility</h2>
            <div class="chart">
                <img src="{{ charts.anticheat }}" alt="Anti-Cheat Chart">
            </div>
        </div>
        {% endif %}

        {% if charts.ml_accuracy %}
        <div class="section">
            <h2>ML Prediction Accuracy</h2>
            <div class="chart">
                <img src="{{ charts.ml_accuracy }}" alt="ML Accuracy Chart">
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>System Information</h2>
            <table>
                {% for key, value in system_info.items() %}
                <tr>
                    <td><strong>{{ key.replace('_', ' ').title() }}</strong></td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>Generated by Steam Deck QA Framework | Session: {{ session_id }}</p>
        </div>
    </div>
</body>
</html>
            """
            
            # Render template
            template = Template(html_template)
            html_content = template.render(**report_data)
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Report Generation Error</h1><p>{str(e)}</p></body></html>"
    
    async def _generate_executive_summary_content(self, summary_data: Dict[str, Any]) -> str:
        """Generate executive summary HTML content"""
        try:
            html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steam Deck QA Framework - Executive Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 3px solid #2196F3; padding-bottom: 20px; }
        .status-banner { text-align: center; padding: 20px; margin: 20px 0; border-radius: 8px; font-size: 1.5em; font-weight: bold; }
        .status-excellent { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
        .status-good { background-color: #cce7ff; color: #0056b3; border: 2px solid #99d3ff; }
        .status-acceptable { background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
        .status-poor { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
        .status-critical { background-color: #dc3545; color: white; border: 2px solid #bd2130; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .metric { text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }
        .metric-value { font-size: 2.5em; font-weight: bold; color: #2196F3; }
        .metric-label { font-size: 1em; color: #666; margin-top: 10px; }
        .section { margin: 30px 0; }
        .section h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .recommendations { background-color: #e9ecef; border-radius: 8px; padding: 20px; }
        .recommendations h3 { margin-top: 0; color: #495057; }
        .recommendations ul { margin: 15px 0; padding-left: 20px; }
        .recommendations li { margin: 8px 0; line-height: 1.4; }
        .critical-issues { background-color: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; padding: 20px; margin: 20px 0; }
        .critical-issues h3 { color: #721c24; margin-top: 0; }
        .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Steam Deck QA Framework</h1>
            <h2>Executive Summary</h2>
            <p>Session ID: {{ session_id }} | Generated: {{ generation_time }}</p>
        </div>

        <div class="status-banner status-{{ overall_status.lower() }}">
            OVERALL STATUS: {{ overall_status }}
        </div>

        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{{ key_metrics.total_games }}</div>
                <div class="metric-label">Total Games Tested</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(pass_rate * 100) }}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ "%.1f"|format(key_metrics.average_fps) }}</div>
                <div class="metric-label">Average FPS</div>
            </div>
            <div class="metric">
                <div class="metric-value">{{ key_metrics.severe_stutters }}</div>
                <div class="metric-label">Severe Stutters</div>
            </div>
        </div>

        {% if critical_issues %}
        <div class="critical-issues">
            <h3>🚨 Critical Issues Requiring Immediate Attention</h3>
            <ul>
            {% for issue in critical_issues %}
                <li>{{ issue }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}

        <div class="section">
            <h2>Key Findings</h2>
            <ul>
                <li><strong>Game Compatibility:</strong> {{ key_metrics.passed_games }} out of {{ key_metrics.total_games }} games passed all tests</li>
                <li><strong>Performance:</strong> Average FPS across all games is {{ "%.1f"|format(key_metrics.average_fps) }}</li>
                <li><strong>Stability:</strong> {{ key_metrics.severe_stutters }} severe stutter events detected</li>
                {% if key_metrics.cache_issues > 0 %}
                <li><strong>Cache Issues:</strong> {{ key_metrics.cache_issues }} games have shader cache problems</li>
                {% endif %}
                {% if key_metrics.anticheat_issues > 0 %}
                <li><strong>Anti-Cheat Issues:</strong> {{ key_metrics.anticheat_issues }} games have anti-cheat compatibility problems</li>
                {% endif %}
            </ul>
        </div>

        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <h3>Action Items</h3>
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>For detailed technical information, refer to the comprehensive report.</p>
            <p>Generated by Steam Deck QA Framework | Session: {{ session_id }}</p>
        </div>
    </div>
</body>
</html>
            """
            
            template = Template(html_template)
            html_content = template.render(**summary_data)
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return f"<html><body><h1>Summary Generation Error</h1><p>{str(e)}</p></body></html>"
    
    async def _generate_regression_report_html(self, report_data: Dict[str, Any]) -> str:
        """Generate regression report HTML content"""
        try:
            html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steam Deck QA Framework - Regression Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 3px solid #2196F3; padding-bottom: 20px; }
        .regression-status { text-align: center; padding: 20px; margin: 20px 0; border-radius: 8px; font-size: 1.5em; font-weight: bold; }
        .no-regression { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
        .regression-detected { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
        .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 30px 0; }
        .comparison-section { padding: 20px; background-color: #f8f9fa; border-radius: 8px; }
        .comparison-section h3 { margin-top: 0; color: #495057; }
        .metric-row { display: flex; justify-content: space-between; margin: 10px 0; padding: 5px 0; border-bottom: 1px solid #dee2e6; }
        .metric-label { font-weight: bold; }
        .metric-value { color: #6c757d; }
        .section { margin: 30px 0; }
        .section h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
        .issues-list { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 15px; }
        .critical-list { background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; padding: 15px; }
        .success-list { background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; padding: 15px; }
        .recommendations { background-color: #e9ecef; border-radius: 8px; padding: 20px; }
        .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Steam Deck QA Framework</h1>
            <h2>Regression Analysis Report</h2>
            <p>Session ID: {{ session_id }} | Generated: {{ generation_time }}</p>
        </div>

        <div class="regression-status {% if regression_analysis.overall_regression %}regression-detected{% else %}no-regression{% endif %}">
            {% if regression_analysis.overall_regression %}
            🚨 REGRESSION DETECTED
            {% else %}
            ✅ NO REGRESSIONS FOUND
            {% endif %}
        </div>

        <div class="comparison">
            <div class="comparison-section">
                <h3>Baseline Results</h3>
                <div class="metric-row">
                    <span class="metric-label">Total Games:</span>
                    <span class="metric-value">{{ baseline_summary.total_games_tested or 0 }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Passed:</span>
                    <span class="metric-value">{{ baseline_summary.passed or 0 }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Failed:</span>
                    <span class="metric-value">{{ baseline_summary.failed or 0 }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Pass Rate:</span>
                    <span class="metric-value">{{ "%.1f"|format((baseline_summary.pass_rate or 0) * 100) }}%</span>
                </div>
            </div>

            <div class="comparison-section">
                <h3>Current Results</h3>
                <div class="metric-row">
                    <span class="metric-label">Total Games:</span>
                    <span class="metric-value">{{ current_summary.total_games_tested or 0 }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Passed:</span>
                    <span class="metric-value">{{ current_summary.passed or 0 }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Failed:</span>
                    <span class="metric-value">{{ current_summary.failed or 0 }}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Pass Rate:</span>
                    <span class="metric-value">{{ "%.1f"|format((current_summary.pass_rate or 0) * 100) }}%</span>
                </div>
            </div>
        </div>

        {% if charts.performance_comparison %}
        <div class="section">
            <h2>Performance Comparison</h2>
            <div class="chart">
                <img src="{{ charts.performance_comparison }}" alt="Performance Comparison Chart">
            </div>
        </div>
        {% endif %}

        {% if charts.regression_impact %}
        <div class="section">
            <h2>Regression Impact Analysis</h2>
            <div class="chart">
                <img src="{{ charts.regression_impact }}" alt="Regression Impact Chart">
            </div>
        </div>
        {% endif %}

        {% if regression_analysis.new_failures %}
        <div class="section">
            <h2>New Failures</h2>
            <div class="critical-list">
                <h3>⚠️ Games that now fail:</h3>
                <ul>
                {% for failure in regression_analysis.new_failures %}
                    <li>{{ failure.replace('_', ' ').title() }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}

        {% if regression_analysis.performance_regressions %}
        <div class="section">
            <h2>Performance Regressions</h2>
            <div class="issues-list">
                <h3>📉 Performance degradation detected:</h3>
                <ul>
                {% for regression in regression_analysis.performance_regressions %}
                    <li>{{ regression.game.replace('_', ' ').title() }}
                    {% if regression.details.regression_percent %}
                        - {{ "%.1f"|format(regression.details.regression_percent) }}% FPS drop
                    {% endif %}
                    </li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}

        {% if regression_analysis.fixed_issues %}
        <div class="section">
            <h2>Fixed Issues</h2>
            <div class="success-list">
                <h3>✅ Games that are now working:</h3>
                <ul>
                {% for fix in regression_analysis.fixed_issues %}
                    <li>{{ fix.replace('_', ' ').title() }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <h2>Recommendations</h2>
            <div class="recommendations">
                <ul>
                {% for rec in recommendations %}
                    <li>{{ rec }}</li>
                {% endfor %}
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>Generated by Steam Deck QA Framework | Session: {{ session_id }}</p>
        </div>
    </div>
</body>
</html>
            """
            
            template = Template(html_template)
            html_content = template.render(**report_data)
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Error generating regression report: {e}")
            return f"<html><body><h1>Regression Report Generation Error</h1><p>{str(e)}</p></body></html>"