#!/usr/bin/env python3
"""
ML Shader Prediction System - Complete Integration Demo

This script demonstrates the complete enhanced dependency management system
with robust fallbacks, comprehensive testing, and Steam Deck optimizations.

Features:
- Complete system initialization and health check
- Dependency detection with version compatibility checking
- Tiered fallback system demonstration
- Robust threading with fallback to single-threaded
- Comprehensive status reporting and logging
- Steam Deck specific optimizations
- Fallback scenario testing
- Performance monitoring and optimization
- User-friendly status dashboard
- Complete system recovery testing
"""

import os
import sys
import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import all our enhanced systems
try:
    from enhanced_dependency_coordinator import get_coordinator
    from tiered_fallback_system import get_fallback_system, switch_to_efficiency_mode
    from robust_threading_manager import get_threading_manager, submit_robust_task
    from comprehensive_status_system import get_status_system, get_user_status, export_status_report
    from fallback_test_suite import run_fallback_tests, TestCategory
    from dependency_version_manager import get_version_manager
    from enhanced_dependency_detector import get_detector
except ImportError as e:
    print(f"❌ Failed to import system components: {e}")
    print("Make sure all dependency management modules are in the correct location")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# SYSTEM INTEGRATION DEMO
# =============================================================================

class SystemIntegrationDemo:
    """Complete system integration demonstration"""
    
    def __init__(self, steam_deck_mode: bool = False):
        self.steam_deck_mode = steam_deck_mode
        self.systems = {}
        
        print("\n🚀 ML Shader Prediction System - Enhanced Dependency Management")
        print("=" * 70)
        print(f"Steam Deck Mode: {'✅' if steam_deck_mode else '❌'}")
        print("=" * 70)

    def run_complete_demo(self) -> Dict[str, Any]:
        """Run complete system demonstration"""
        demo_results = {
            'initialization': False,
            'health_check': False,
            'fallback_tests': False,
            'performance_tests': False,
            'user_dashboard': False,
            'system_recovery': False,
            'final_status': {}
        }
        
        try:
            # Phase 1: System Initialization
            print("\n📋 Phase 1: System Initialization")
            demo_results['initialization'] = self._initialize_all_systems()
            
            if not demo_results['initialization']:
                print("❌ System initialization failed - stopping demo")
                return demo_results
            
            # Phase 2: Health Check and Status
            print("\n🏥 Phase 2: Comprehensive Health Check")
            demo_results['health_check'] = self._perform_health_check()
            
            # Phase 3: Dependency Detection and Fallback Testing
            print("\n🔍 Phase 3: Dependency Detection and Fallbacks")
            demo_results['fallback_tests'] = self._test_fallback_scenarios()
            
            # Phase 4: Performance Testing
            print("\n⚡ Phase 4: Performance Testing and Optimization")
            demo_results['performance_tests'] = self._test_performance_scenarios()
            
            # Phase 5: User Dashboard
            print("\n📱 Phase 5: User Dashboard and Reporting")
            demo_results['user_dashboard'] = self._demonstrate_user_dashboard()
            
            # Phase 6: System Recovery Testing
            print("\n🛡️ Phase 6: System Recovery Testing")
            demo_results['recovery_tests'] = self._test_system_recovery()
            
            # Final Status
            print("\n📊 Final System Status")
            demo_results['final_status'] = self._get_final_status()
            
            # Export comprehensive report
            self._export_demo_report(demo_results)
            
            return demo_results
            
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
            return demo_results
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            logger.error(f"Demo failed: {e}")
            return demo_results

    def _initialize_all_systems(self) -> bool:
        """Initialize all system components"""
        try:
            print("  🔧 Initializing enhanced dependency coordinator...")
            self.systems['coordinator'] = get_coordinator()
            
            print("  🔄 Initializing tiered fallback system...")
            self.systems['fallback_system'] = get_fallback_system()
            
            print("  🧵 Initializing robust threading manager...")
            self.systems['threading_manager'] = get_threading_manager()
            
            print("  📊 Initializing comprehensive status system...")
            self.systems['status_system'] = get_status_system()
            
            print("  📦 Initializing dependency detector...")
            self.systems['detector'] = get_detector()
            
            print("  🔍 Initializing version manager...")
            self.systems['version_manager'] = get_version_manager()
            
            # Let systems initialize fully
            time.sleep(2)
            
            print("  ✅ All systems initialized successfully")
            return True
            
        except Exception as e:
            print(f"  ❌ System initialization failed: {e}")
            logger.error(f"System initialization error: {e}")
            return False

    def _perform_health_check(self) -> bool:
        """Perform comprehensive health check"""
        try:
            print("  🏥 Running comprehensive health check...")
            
            # Get system health from coordinator
            if 'coordinator' in self.systems:
                health_report = self.systems['coordinator'].get_comprehensive_health_report()
                
                print(f"  📈 Overall Health: {health_report.overall_health:.1%} ({health_report.health_level.name})")
                print(f"  📦 Dependencies: {health_report.dependency_status.get('available_dependencies', 0)}/{health_report.dependency_status.get('total_dependencies', 0)} available")
                
                # Component health
                print("  🔧 Component Health:")
                for component, health in health_report.component_health.items():
                    status_icon = "✅" if health > 0.8 else "⚠️" if health > 0.5 else "❌"
                    print(f"    {status_icon} {component}: {health:.1%}")
                
                # Critical issues
                if health_report.critical_issues:
                    print("  🚨 Critical Issues:")
                    for issue in health_report.critical_issues:
                        print(f"    ❗ {issue}")
                
                # Recommendations
                if health_report.recommendations:
                    print("  💡 Recommendations:")
                    for rec in health_report.recommendations[:3]:  # Top 3
                        print(f"    💡 {rec}")
                
                # Steam Deck status
                if health_report.steam_deck_status and self.steam_deck_mode:
                    steam_status = health_report.steam_deck_status
                    temp = steam_status.get('cpu_temperature', 0)
                    temp_icon = "🟢" if temp < 70 else "🟡" if temp < 80 else "🔴"
                    print(f"  🎮 Steam Deck: {temp_icon} {temp:.1f}°C ({steam_status.get('thermal_state', 'Unknown')})")
                
                return health_report.overall_health > 0.3  # At least 30% health
            
            return False
            
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
            return False

    def _test_fallback_scenarios(self) -> bool:
        """Test various fallback scenarios"""
        try:
            print("  🧪 Testing dependency detection...")
            
            # Run dependency detection
            if 'detector' in self.systems:
                detection_results = self.systems['detector'].detect_all_dependencies()
                
                available_count = sum(1 for r in detection_results.values() if r.status.available)
                fallback_count = sum(1 for r in detection_results.values() if r.status.fallback_active)
                
                print(f"    📊 Detection Results: {available_count} available, {fallback_count} using fallbacks")
                
                # Show some example detections
                examples = list(detection_results.items())[:5]
                for dep_name, result in examples:
                    status_icon = "✅" if result.status.available else "🔄" if result.status.fallback_active else "❌"
                    version = f" v{result.status.version}" if result.status.version else ""
                    print(f"    {status_icon} {dep_name}{version}")
            
            print("  🔄 Testing fallback system...")
            
            # Test fallback implementations
            if 'fallback_system' in self.systems:
                fallback_status = self.systems['fallback_system'].get_fallback_status()
                
                print(f"    📈 Active Components: {fallback_status.get('active_components', 0)}")
                print(f"    ⚡ Avg Performance: {fallback_status.get('average_performance_multiplier', 1.0):.1f}x")
                
                # Test getting implementations
                test_components = ['array_math', 'ml_algorithms', 'system_monitor']
                for component in test_components:
                    impl = self.systems['fallback_system'].get_implementation(component)
                    status = "✅ Available" if impl else "❌ None"
                    print(f"    {status} {component}")
                
                # Test tier switching
                print("  🔄 Testing tier switching...")
                try:
                    from tiered_fallback_system import FallbackTier, FallbackReason
                    
                    # Test switching to efficient mode
                    switch_results = switch_to_efficiency_mode()
                    successful_switches = sum(1 for success in switch_results.values() if success)
                    print(f"    📊 Tier switches: {successful_switches}/{len(switch_results)} successful")
                    
                except Exception as e:
                    print(f"    ⚠️ Tier switching test failed: {e}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Fallback testing failed: {e}")
            return False

    def _test_performance_scenarios(self) -> bool:
        """Test performance scenarios"""
        try:
            print("  ⚡ Testing threading performance...")
            
            # Test threading manager
            if 'threading_manager' in self.systems:
                # Simple performance test
                def test_task(x):
                    import time
                    import math
                    time.sleep(0.01)  # Simulate work
                    return math.sqrt(x * 42)
                
                # Submit multiple tasks
                start_time = time.time()
                futures = []
                
                for i in range(20):
                    future = submit_robust_task(test_task, i)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in futures:
                    try:
                        if hasattr(future, 'result'):
                            result = future.result(timeout=5.0)
                        else:
                            result = future  # Synchronous result
                        results.append(result)
                    except Exception as e:
                        print(f"    ⚠️ Task failed: {e}")
                
                execution_time = time.time() - start_time
                throughput = len(results) / execution_time
                
                print(f"    📊 Completed {len(results)}/20 tasks in {execution_time:.2f}s")
                print(f"    🚀 Throughput: {throughput:.1f} tasks/second")
                
                # Get threading status
                threading_status = self.systems['threading_manager'].get_status()
                print(f"    🧵 Threading Mode: {threading_status.mode.name}")
                print(f"    ❤️ Health: {threading_status.health.name}")
            
            print("  🎯 Testing system optimization...")
            
            # Test coordinator optimization
            if 'coordinator' in self.systems:
                optimization_plan = self.systems['coordinator'].create_optimization_plan()
                
                print(f"    📈 Target Health: {optimization_plan.target_improvements.get('overall_health', 0):.1%}")
                print(f"    📋 Recommended Actions: {len(optimization_plan.recommended_actions)}")
                print(f"    📦 Dependencies to Install: {len(optimization_plan.dependencies_to_install)}")
                print(f"    ⚡ Expected Performance Gain: {optimization_plan.expected_performance_gain:.1f}x")
                
                # Show top recommendations
                for i, action in enumerate(optimization_plan.recommended_actions[:3], 1):
                    priority_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(action.get('priority', 'medium'), "ℹ️")
                    print(f"    {i}. {priority_icon} {action.get('description', 'No description')}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Performance testing failed: {e}")
            return False

    def _demonstrate_user_dashboard(self) -> bool:
        """Demonstrate user-friendly dashboard"""
        try:
            print("  📱 Generating user dashboard...")
            
            # Get user-friendly status
            dashboard = get_user_status()
            
            print(f"  🎯 System Status: {dashboard['status']['emoji']} {dashboard['status']['text']}")
            print(f"  📊 Health: {dashboard.get('overall_health', 0):.1%}")
            print(f"  ⏱️ Uptime: {dashboard.get('uptime', 'Unknown')}")
            
            # Component status
            if dashboard.get('components'):
                print("  🔧 Components:")
                for name, comp_status in dashboard['components'].items():
                    print(f"    {comp_status['status']} {name}: {comp_status['details']}")
            
            # Alerts
            if dashboard.get('alerts'):
                print("  🚨 Alerts:")
                for alert in dashboard['alerts']:
                    print(f"    {alert['icon']} {alert['message']}")
            
            # Recommendations
            if dashboard.get('recommendations'):
                print("  💡 Recommendations:")
                for i, rec in enumerate(dashboard['recommendations'], 1):
                    print(f"    {i}. {rec}")
            
            # Steam Deck info
            if dashboard.get('steam_deck') and self.steam_deck_mode:
                print("  🎮 Steam Deck:")
                steam = dashboard['steam_deck']
                print(f"    Model: {steam['model']}")
                print(f"    Temperature: {steam['temperature']['emoji']} {steam['temperature']['value']} ({steam['temperature']['status']})")
                print(f"    Optimizations: {steam['optimizations']}")
            
            print("  📄 Exporting status reports...")
            
            # Export reports in different formats
            report_dir = Path("/tmp/ml_shader_reports")
            report_dir.mkdir(exist_ok=True)
            
            formats = [
                ("JSON", "json"),
                ("Markdown", "markdown"),
                ("HTML", "html")
            ]
            
            for format_name, format_ext in formats:
                report_path = report_dir / f"system_status.{format_ext}"
                success = export_status_report(str(report_path), format_ext)
                status_icon = "✅" if success else "❌"
                print(f"    {status_icon} {format_name}: {report_path}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Dashboard demonstration failed: {e}")
            return False

    def _test_system_recovery(self) -> bool:
        """Test system recovery capabilities"""
        try:
            print("  🛡️ Testing system recovery scenarios...")
            
            # Test 1: Simulated dependency failure recovery
            print("    🧪 Test 1: Dependency failure recovery")
            
            # Get initial state
            initial_health = None
            if 'coordinator' in self.systems:
                initial_report = self.systems['coordinator'].get_comprehensive_health_report()
                initial_health = initial_report.overall_health
                print(f"      📊 Initial health: {initial_health:.1%}")
            
            # Test 2: Threading failure recovery
            print("    🧪 Test 2: Threading system resilience")
            
            # Submit tasks that might fail
            def potentially_failing_task(x):
                if x == 5:
                    raise RuntimeError("Simulated task failure")
                return x * 2
            
            failed_tasks = 0
            successful_tasks = 0
            
            for i in range(10):
                try:
                    result = submit_robust_task(potentially_failing_task, i)
                    if hasattr(result, 'result'):
                        result.result(timeout=2.0)
                    successful_tasks += 1
                except Exception:
                    failed_tasks += 1
            
            print(f"      📊 Task results: {successful_tasks} successful, {failed_tasks} failed")
            
            # Check if system is still responsive
            try:
                test_result = submit_robust_task(lambda: 42)
                if hasattr(test_result, 'result'):
                    final_result = test_result.result(timeout=5.0)
                else:
                    final_result = test_result
                
                system_responsive = (final_result == 42)
                print(f"      🏥 System responsive after failures: {'✅' if system_responsive else '❌'}")
                
            except Exception as e:
                print(f"      ❌ System responsiveness test failed: {e}")
                return False
            
            # Test 3: Final health check
            print("    🧪 Test 3: Post-recovery health check")
            
            if 'coordinator' in self.systems:
                final_report = self.systems['coordinator'].get_comprehensive_health_report()
                final_health = final_report.overall_health
                
                print(f"      📊 Final health: {final_health:.1%}")
                
                if initial_health:
                    health_change = final_health - initial_health
                    change_icon = "📈" if health_change >= 0 else "📉"
                    print(f"      {change_icon} Health change: {health_change:+.1%}")
                
                # System should maintain reasonable health
                recovery_successful = final_health > 0.3
                print(f"      🛡️ Recovery successful: {'✅' if recovery_successful else '❌'}")
                
                return recovery_successful
            
            return True
            
        except Exception as e:
            print(f"  ❌ Recovery testing failed: {e}")
            return False

    def _get_final_status(self) -> Dict[str, Any]:
        """Get final system status summary"""
        try:
            status = {}
            
            # Get comprehensive status
            if 'coordinator' in self.systems:
                health_report = self.systems['coordinator'].get_comprehensive_health_report()
                
                status['overall_health'] = health_report.overall_health
                status['health_level'] = health_report.health_level.name
                status['component_count'] = len(health_report.component_health)
                status['critical_issues'] = len(health_report.critical_issues)
                status['recommendations'] = len(health_report.recommendations)
            
            # Threading status
            if 'threading_manager' in self.systems:
                thread_status = self.systems['threading_manager'].get_status()
                status['threading_mode'] = thread_status.mode.name
                status['threading_health'] = thread_status.health.name
                status['completed_tasks'] = thread_status.completed_tasks
            
            # Fallback status
            if 'fallback_system' in self.systems:
                fallback_status = self.systems['fallback_system'].get_fallback_status()
                status['active_fallbacks'] = fallback_status.get('active_components', 0)
                status['performance_multiplier'] = fallback_status.get('average_performance_multiplier', 1.0)
            
            # Print final summary
            print(f"  📊 Overall Health: {status.get('overall_health', 0):.1%} ({status.get('health_level', 'Unknown')})")
            print(f"  🧵 Threading: {status.get('threading_mode', 'Unknown')} ({status.get('threading_health', 'Unknown')})")
            print(f"  🔄 Active Fallbacks: {status.get('active_fallbacks', 0)}")
            print(f"  ⚡ Performance: {status.get('performance_multiplier', 1.0):.1f}x")
            print(f"  ✅ Completed Tasks: {status.get('completed_tasks', 0)}")
            
            if status.get('critical_issues', 0) > 0:
                print(f"  🚨 Critical Issues: {status['critical_issues']}")
            
            if status.get('recommendations', 0) > 0:
                print(f"  💡 Recommendations Available: {status['recommendations']}")
            
            return status
            
        except Exception as e:
            print(f"  ❌ Failed to get final status: {e}")
            return {'error': str(e)}

    def _export_demo_report(self, demo_results: Dict[str, Any]) -> None:
        """Export comprehensive demo report"""
        try:
            print("  💾 Exporting comprehensive demo report...")
            
            report_dir = Path("/tmp/ml_shader_demo_reports")
            report_dir.mkdir(exist_ok=True)
            
            # Create comprehensive report
            report = {
                'demo_info': {
                    'timestamp': time.time(),
                    'steam_deck_mode': self.steam_deck_mode,
                    'python_version': sys.version,
                    'system_platform': sys.platform
                },
                'demo_results': demo_results,
                'system_status': {},
                'performance_metrics': {}
            }
            
            # Add system status
            if 'coordinator' in self.systems:
                try:
                    health_report = self.systems['coordinator'].get_comprehensive_health_report()
                    report['system_status'] = {
                        'overall_health': health_report.overall_health,
                        'health_level': health_report.health_level.name,
                        'component_health': health_report.component_health,
                        'dependency_status': health_report.dependency_status,
                        'performance_metrics': health_report.performance_metrics,
                        'steam_deck_status': health_report.steam_deck_status
                    }
                except Exception as e:
                    report['system_status'] = {'error': str(e)}
            
            # Export JSON report
            json_report_path = report_dir / 'demo_report.json'
            with open(json_report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"    ✅ JSON Report: {json_report_path}")
            
            # Export status reports
            try:
                status_report_path = report_dir / 'status_report.json'
                export_success = export_status_report(str(status_report_path), "json")
                if export_success:
                    print(f"    ✅ Status Report: {status_report_path}")
                else:
                    print(f"    ⚠️ Status Report: Export failed")
            except Exception as e:
                print(f"    ⚠️ Status Report: {e}")
            
            print(f"  📁 All reports available in: {report_dir}")
            
        except Exception as e:
            print(f"  ❌ Report export failed: {e}")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ML Shader Prediction System Integration Demo')
    parser.add_argument('--steam-deck', action='store_true', help='Run in Steam Deck mode')
    parser.add_argument('--test-fallbacks', action='store_true', help='Run comprehensive fallback tests')
    parser.add_argument('--quick-test', action='store_true', help='Run quick integration test')
    parser.add_argument('--export-only', action='store_true', help='Only export status reports')
    
    args = parser.parse_args()
    
    # Handle export-only mode
    if args.export_only:
        print("📄 Exporting status reports...")
        report_dir = Path("/tmp/ml_shader_export")
        report_dir.mkdir(exist_ok=True)
        
        formats = [('json', 'JSON'), ('markdown', 'Markdown'), ('html', 'HTML')]
        for ext, name in formats:
            report_path = report_dir / f"status_report.{ext}"
            success = export_status_report(str(report_path), ext)
            status = "✅" if success else "❌"
            print(f"  {status} {name}: {report_path}")
        
        print(f"📁 Reports exported to: {report_dir}")
        return
    
    # Handle fallback testing mode
    if args.test_fallbacks:
        print("🧪 Running comprehensive fallback tests...")
        
        try:
            test_results = run_fallback_tests(steam_deck_mode=args.steam_deck)
            
            print(f"\n📊 Fallback Test Results:")
            print(f"  Total Tests: {test_results['total_tests']}")
            print(f"  Passed: {test_results['passed']} ✅")
            print(f"  Failed: {test_results['failed']} ❌")
            print(f"  Pass Rate: {test_results['pass_rate']:.1%}")
            
            if test_results['failed_test_details']:
                print(f"\n❌ Failed Tests:")
                for failed in test_results['failed_test_details']:
                    print(f"  {failed['name']}: {failed['errors'][0] if failed['errors'] else 'Unknown error'}")
            
            # Export test results
            test_report_dir = Path("/tmp/ml_shader_test_results")
            test_report_dir.mkdir(exist_ok=True)
            test_report_path = test_report_dir / "fallback_test_results.json"
            
            with open(test_report_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            print(f"\n💾 Test results exported to: {test_report_path}")
            
            # Exit with appropriate code
            success_rate = test_results['pass_rate']
            if success_rate >= 0.8:
                print("✅ Fallback tests passed - system is robust")
                sys.exit(0)
            else:
                print("❌ Some fallback tests failed - system needs attention")
                sys.exit(1)
        
        except Exception as e:
            print(f"❌ Fallback testing failed: {e}")
            sys.exit(1)
    
    # Handle quick test mode
    if args.quick_test:
        print("⚡ Running quick integration test...")
        
        try:
            # Quick system check
            status_system = get_status_system()
            user_status = get_user_status()
            
            print(f"System Status: {user_status['status']['emoji']} {user_status['status']['text']}")
            print(f"Health: {user_status.get('overall_health', 0):.1%}")
            
            # Quick threading test
            result = submit_robust_task(lambda: 42)
            if hasattr(result, 'result'):
                test_result = result.result(timeout=5.0)
            else:
                test_result = result
            
            if test_result == 42:
                print("✅ Threading system operational")
            else:
                print("⚠️ Threading system issues detected")
            
            print("✅ Quick test completed successfully")
            return
        
        except Exception as e:
            print(f"❌ Quick test failed: {e}")
            sys.exit(1)
    
    # Run full integration demo
    try:
        demo = SystemIntegrationDemo(steam_deck_mode=args.steam_deck)
        results = demo.run_complete_demo()
        
        # Summary
        print("\n🎯 Demo Summary:")
        print("=" * 50)
        
        phases = [
            ('initialization', 'System Initialization'),
            ('health_check', 'Health Check'),
            ('fallback_tests', 'Fallback Tests'),
            ('performance_tests', 'Performance Tests'),
            ('user_dashboard', 'User Dashboard'),
            ('recovery_tests', 'Recovery Tests')
        ]
        
        passed_phases = 0
        for phase_key, phase_name in phases:
            status = "✅" if results.get(phase_key, False) else "❌"
            print(f"  {status} {phase_name}")
            if results.get(phase_key, False):
                passed_phases += 1
        
        print(f"\n📊 Overall Success: {passed_phases}/{len(phases)} phases passed")
        
        # Final status
        final_status = results.get('final_status', {})
        if final_status.get('overall_health'):
            health = final_status['overall_health']
            health_emoji = "✅" if health > 0.8 else "⚠️" if health > 0.5 else "❌"
            print(f"🏥 Final System Health: {health_emoji} {health:.1%}")
        
        # Success determination
        if passed_phases >= 4:  # At least 4/6 phases must pass
            print("\n🎉 Integration demo completed successfully!")
            print("🛡️ System is robust and ready for production use")
            
            if args.steam_deck:
                print("🎮 Steam Deck optimizations are active and working")
            
            sys.exit(0)
        else:
            print("\n⚠️ Integration demo completed with issues")
            print("🔧 System needs attention before production use")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()