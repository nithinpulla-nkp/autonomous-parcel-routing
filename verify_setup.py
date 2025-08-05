#!/usr/bin/env python3
"""
Quick setup verification script to ensure all imports work correctly.
Run this to test your APR package installation.
"""

def verify_apr_imports():
    """Test that all APR modules can be imported correctly."""
    try:
        # Core package
        import apr
        print(f"✅ apr package (v{apr.__version__})")
        
        # Environment
        from apr import WarehouseEnv
        print("✅ WarehouseEnv")
        
        # Logger
        from apr import RunLogger  
        print("✅ RunLogger")
        
        # Agents
        from apr.agents import BaseAgent, QLearningAgent, SarsaAgent, create_agent
        print("✅ BaseAgent, QLearningAgent, SarsaAgent, create_agent")
        
        # Training module
        from apr import train
        print("✅ train module")
        
        print("\n🎉 All imports successful! Package is properly configured.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Try running: pip install -e .")
        return False

def verify_file_structure():
    """Check that all expected files and directories exist."""
    from pathlib import Path
    
    root = Path(__file__).parent
    expected_files = [
        "src/apr/__init__.py",
        "src/apr/env.py", 
        "src/apr/agents/__init__.py",
        "src/apr/agents/q_learning.py",
        "cfg/baseline.yaml",
        "pyproject.toml"
    ]
    
    print("\n📁 File structure check:")
    all_good = True
    for file_path in expected_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing!")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🔍 Verifying APR package setup...\n")
    
    imports_ok = verify_apr_imports()
    files_ok = verify_file_structure()
    
    if imports_ok and files_ok:
        print("\n🚀 Setup verification complete - everything looks good!")
    else:
        print("\n⚠️  Some issues found - please fix before proceeding.")