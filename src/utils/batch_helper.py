from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from .config import Config

@dataclass
class BatchHelper:
    config_path: Path
    
    def get_current_batch(self) -> Tuple[int, int]:
        """Get current batch start and end."""
        config = Config(self.config_path)
        return (
            config.get("batch.current_start", 1),
            config.get("batch.current_end", 5)
        )
    
    def update_batch_range(self, start: int, end: int) -> None:
        """Update batch range in config."""
        import yaml
        
        with open(self.config_path) as f:
            config_data = yaml.safe_load(f)
        
        if 'batch' not in config_data:
            config_data['batch'] = {}
            
        config_data['batch']['current_start'] = start
        config_data['batch']['current_end'] = end
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    def next_batch(self) -> Tuple[int, int]:
        """Get next batch range and update config."""
        config = Config(self.config_path)
        step = config.get("batch.step_size", 5)
        total = config.get("batch.total_parts", 200)
        
        current_start, current_end = self.get_current_batch()
        next_start = current_end + 1
        next_end = min(next_start + step - 1, total)
        
        if next_start > total:
            raise ValueError("No more batches available")
            
        self.update_batch_range(next_start, next_end)
        return next_start, next_end