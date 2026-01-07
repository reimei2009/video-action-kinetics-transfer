"""
Test entrypoints - Kiểm tra các script có chạy được không
"""

import sys
from pathlib import Path

def test_import(module_path):
    """Test import Python file"""
    try:
        # Compile để check syntax
        with open(module_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, module_path, 'exec')
        print(f"✓ {module_path} - Syntax OK")
        return True
    except Exception as e:
        print(f"✗ {module_path} - Error: {e}")
        return False

def main():
    print("=== Testing Project Structure ===\n")
    
    files_to_test = [
        'src/__init__.py',
        'src/models/__init__.py',
        'src/models/x3d_wrapper.py',
        'src/datasets/__init__.py',
        'src/datasets/kinetics_subset.py',
        'src/datasets/nsar_sports.py',
        'src/train_kinetics.py',
        'src/train_nsar.py',
        'src/inference.py',
    ]
    
    all_passed = True
    for file_path in files_to_test:
        if Path(file_path).exists():
            if not test_import(file_path):
                all_passed = False
        else:
            print(f"⚠ {file_path} - File not found")
            all_passed = False
    
    print("\n=== Checking Configs ===\n")
    configs = [
        'configs/kinetics_subset.yaml',
        'configs/nsar_transfer.yaml',
    ]
    
    for config in configs:
        if Path(config).exists():
            print(f"✓ {config} - Exists")
        else:
            print(f"✗ {config} - Not found")
            all_passed = False
    
    print("\n=== Checking Directories ===\n")
    dirs = ['weights', 'scripts', 'configs', 'src/datasets', 'src/models']
    for dir_path in dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/ - Exists")
        else:
            print(f"✗ {dir_path}/ - Not found")
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ All tests passed! Project structure is ready.")
        print("Next step: Push to GitHub")
    else:
        print("⚠ Some tests failed. Please check the files above.")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
