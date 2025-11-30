#!/usr/bin/env python3
"""
Fix GPT-OSS model config.json to use gpt_neox instead of gpt_oss
so transformers can load it.
"""

import json
import sys
from pathlib import Path

def fix_model_config(model_path):
    """Change model_type from gpt_oss to gpt_neox in config.json"""
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    
    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        return False
    
    print(f"Reading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if config.get('model_type') == 'gpt_oss':
        print(f"Found model_type: gpt_oss")
        print("Changing to: gpt_neox")
        config['model_type'] = 'gpt_neox'
        
        # Backup original
        backup_path = config_path.with_suffix('.json.bak')
        print(f"Creating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Write modified config
        print(f"Writing modified config to {config_path}...")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ Config fixed successfully!")
        return True
    elif config.get('model_type') == 'gpt_neox':
        print("✓ Config already has model_type: gpt_neox (no changes needed)")
        return True
    else:
        print(f"Warning: model_type is '{config.get('model_type')}', not 'gpt_oss'")
        print("Proceeding anyway...")
        config['model_type'] = 'gpt_neox'
        
        backup_path = config_path.with_suffix('.json.bak')
        print(f"Creating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ Config updated!")
        return True

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/media/fmodels/TheHouseOfTheDude/gpt-oss-20b_uncanned"
    success = fix_model_config(model_path)
    sys.exit(0 if success else 1)

