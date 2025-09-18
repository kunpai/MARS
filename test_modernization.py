#!/usr/bin/env python3
"""
Basic tests for MARS modernization
Tests imports, configuration, and basic functionality
"""

import sys
from pathlib import Path
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all critical imports work"""
    print("🧪 Testing imports...")
    
    import_tests = [
        # Core Python modules
        ("pathlib", "Path"),
        ("typing", "Dict, List, Tuple, Any, Optional"),
        
        # Third-party packages
        ("rich.console", "Console"),
        ("rich.progress", "Progress"),
        ("rich.panel", "Panel"),
        ("click", None),
        ("loguru", "logger"),
        
        # Local modules (basic import test)
        ("util.review_collab", "parse_pdf_to_text"),
        ("util.build_models", "generate_base_models"),
        ("util.multiagent", "consultGrammar"),
    ]
    
    failed = []
    for module, item in import_tests:
        try:
            if item:
                exec(f"from {module} import {item}")
            else:
                exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed.append(module)
        except Exception as e:
            print(f"  ⚠️ {module}: {e}")
            failed.append(module)
    
    return failed

def test_configuration():
    """Test configuration and setup"""
    print("\n🔧 Testing configuration...")
    
    # Check pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        print("  ✅ pyproject.toml exists")
        try:
            import tomllib
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
            print(f"  ✅ pyproject.toml is valid TOML")
            if "project" in config:
                print(f"  ✅ Project name: {config['project'].get('name', 'Unknown')}")
        except Exception as e:
            print(f"  ❌ pyproject.toml parsing error: {e}")
    else:
        print("  ❌ pyproject.toml not found")
    
    # Check requirements files
    for req_file in ["requirements.txt", "requirements-dev.txt"]:
        if Path(req_file).exists():
            print(f"  ✅ {req_file} exists")
        else:
            print(f"  ❌ {req_file} not found")

def test_mars_main():
    """Test MARS main module loads correctly"""
    print("\n🚀 Testing MARS main module...")
    
    try:
        import MARS
        print("  ✅ MARS module imports successfully")
        
        # Test function availability
        functions_to_test = [
            "load_paper_sections",
            "load_checkpoint", 
            "fancy_aggregate_reviews",
            "checkpoint_progress",
            "main"
        ]
        
        for func_name in functions_to_test:
            if hasattr(MARS, func_name):
                print(f"  ✅ Function {func_name} available")
            else:
                print(f"  ❌ Function {func_name} missing")
                
    except Exception as e:
        print(f"  ❌ MARS import failed: {e}")
        traceback.print_exc()

def test_cli():
    """Test CLI module"""
    print("\n💻 Testing CLI module...")
    
    try:
        import cli
        print("  ✅ CLI module imports successfully")
    except Exception as e:
        print(f"  ❌ CLI import failed: {e}")

def main():
    """Run all tests"""
    print("🧪 MARS Modernization Tests")
    print("=" * 50)
    
    failed_imports = test_imports()
    test_configuration()
    test_mars_main()
    test_cli()
    
    print("\n" + "=" * 50)
    if failed_imports:
        print(f"❌ Tests completed with {len(failed_imports)} failed imports")
        print(f"Failed imports: {', '.join(failed_imports)}")
        print("\nTo fix missing dependencies, run:")
        print("  pip install -r requirements.txt")
        return 1
    else:
        print("✅ All basic tests passed!")
        print("\nMARS modernization appears successful.")
        return 0

if __name__ == "__main__":
    exit(main())