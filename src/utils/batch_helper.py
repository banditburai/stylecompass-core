from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from .config import Config
import yaml

@dataclass
class BatchHelper:
    config_path: Path
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def get_current_batch(self) -> Tuple[int, int]:
        """Get current batch start and end from config"""
        return (
            self.config["batch"]["current_start"],
            self.config["batch"]["current_end"]
        )
    
    def next_batch(self, update_config: bool = True) -> Tuple[int, int]:
        """Get next batch range and optionally update config file"""
        current_start = self.config["batch"]["current_start"]
        step_size = self.config["batch"]["step_size"]
        total_parts = self.config["batch"]["total_parts"]
        
        next_start = current_start + step_size
        next_end = min(next_start + step_size - 1, total_parts)
        
        if update_config:
            # Update config in memory
            self.config["batch"]["current_start"] = next_start
            self.config["batch"]["current_end"] = next_end
            
            # Write back to file
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        
        return next_start, next_end