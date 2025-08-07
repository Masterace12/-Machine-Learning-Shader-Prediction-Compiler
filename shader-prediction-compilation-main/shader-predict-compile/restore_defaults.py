#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from settings_manager import get_settings_manager

def main():
    parser = argparse.ArgumentParser(
        description='Restore Shader Predictive Compiler settings to defaults',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Reset settings only
  %(prog)s --cache            # Reset settings and clear cache
  %(prog)s --logs             # Reset settings and clear logs  
  %(prog)s --cache --logs     # Reset everything
  %(prog)s --all              # Reset everything (same as --cache --logs)
        """
    )
    
    parser.add_argument(
        '--cache', 
        action='store_true',
        help='Also clear shader cache directories'
    )
    
    parser.add_argument(
        '--logs', 
        action='store_true',
        help='Also clear log files'
    )
    
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Reset everything (settings, cache, and logs)'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Handle --all flag
    if args.all:
        args.cache = True
        args.logs = True
    
    # Show what will be done
    operations = ["Reset settings to defaults"]
    if args.cache:
        operations.append("Clear shader cache")
    if args.logs:
        operations.append("Clear log files")
    
    if not args.quiet:
        print("🔄 Shader Predictive Compiler - Restore to Defaults")
        print("=" * 60)
        print("The following operations will be performed:")
        for i, op in enumerate(operations, 1):
            print(f"  {i}. {op}")
        print()
        print("✓ A backup will be created before resetting")
        print()
    
    # Confirmation
    if not args.yes:
        response = input("Do you want to continue? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Operation cancelled.")
            return 0
    
    # Perform restore
    try:
        if not args.quiet:
            print("🔄 Performing restore...")
        
        settings_manager = get_settings_manager()
        results = settings_manager.restore_to_defaults(
            include_cache=args.cache,
            include_logs=args.logs
        )
        
        # Show results
        if not args.quiet:
            print("\n📊 Results:")
            
            if results['backup_created']:
                print("  ✓ Settings backup created")
            else:
                print("  ⚠ Could not create backup")
            
            if results['settings_reset']:
                print("  ✓ Settings reset to defaults")
            else:
                print("  ❌ Failed to reset settings")
            
            if results['enhanced_settings_reset']:
                print("  ✓ Enhanced settings reset")
            
            if args.cache:
                if results['cache_cleared']:
                    print("  ✓ Cache cleared")
                else:
                    print("  ⚠ Could not clear all cache files")
            
            if args.logs:
                if results['logs_cleared']:
                    print("  ✓ Logs cleared")
                else:
                    print("  ⚠ Could not clear all log files")
        
        # Summary
        successful_ops = sum(1 for result in results.values() if result)
        total_ops = len([k for k in results.keys() if k != 'backup_created' or args.cache or args.logs])
        
        if not args.quiet:
            print(f"\n🎉 Restore completed: {successful_ops}/{total_ops} operations successful")
            
            if all(results[key] for key in ['settings_reset']):
                print("✓ All essential operations completed successfully")
                return 0
            else:
                print("⚠ Some operations had issues (see above)")
                return 1
        else:
            # Quiet mode: just return success/failure
            return 0 if results['settings_reset'] else 1
            
    except Exception as e:
        if not args.quiet:
            print(f"\n❌ Error during restore: {e}")
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':
    exit(main())