"""
Utility functions for YAML serialization and file operations.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

def save_state_to_yaml(state_dict: Dict[str, Any], data_dir: str, filename: str) -> None:
    """Save a state dictionary to a YAML file with custom formatting.

    Args:
        state_dict: The dictionary to save
        data_dir: The directory where the file should be saved
        filename: The name of the file to save (without extension)
    """
    # Add custom representer for strings
    def str_presenter(dumper, data):
        # Strip whitespace from the string
        data = data.strip()
        # For multiline strings, use literal style
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        # For long strings without newlines, use the folded style
        if len(data) > 80:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='>')
        # For short strings, use plain style
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='')

    yaml.add_representer(str, str_presenter)

    # Add custom representer for None values
    def none_representer(dumper, _):
        return dumper.represent_scalar('tag:yaml.org,2002:null', '')

    yaml.add_representer(type(None), none_representer)

    # Create the full file path
    file_path = Path(data_dir) / f"{filename}.yaml"

    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the state to YAML
    with open(file_path, "w", encoding='utf-8') as f:
        yaml.dump(
            state_dict,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
            default_style=None,  # This prevents unnecessary escaping
            indent=2  # Add consistent indentation
        )
