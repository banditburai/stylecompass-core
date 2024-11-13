import pytest
from pathlib import Path
from src.utils.config import Config

def test_config_loading(tmp_path):
    # Create a temporary config file
    config_content = """
    test_key: test_value
    nested:
        key: value
    """
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    
    # Test loading
    config = Config(config_path)
    assert config.get("test_key") == "test_value"
    assert config.get("nested")["key"] == "value"
    
    # Test default value
    assert config.get("non_existent", "default") == "default"