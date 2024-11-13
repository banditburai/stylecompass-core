import click
from pathlib import Path
from huggingface_hub import scan_cache_dir
from huggingface_hub.utils import HFCacheInfo
from src.utils import setup_logger, Config

logger = setup_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/cache_management.yaml',
              help='Path to config file')
@click.option('--force', is_flag=True, help='Skip deletion confirmation')
def main(config: str, force: bool):
    """Delete the complete HuggingFace cache"""
    config = Config(Path(config))
    
    logger.info("Scanning cache...")
    cache_info: HFCacheInfo = scan_cache_dir()
    
    initial_size_gb = cache_info.size_on_disk / (1024**3)
    logger.info(f"Current cache size: {initial_size_gb:.2f}GB")
    
    revision_hashes = [
        revision.commit_hash
        for repo in cache_info.repos
        for revision in repo.revisions
    ]
    
    if not revision_hashes:
        logger.info("No revisions found in cache.")
        return
    
    strategy = cache_info.delete_revisions(*revision_hashes)
    logger.info(f"Will free {strategy.expected_freed_size_str}")
    
    if not force and config.get("huggingface.cache.confirm_deletion", True):
        if input("Do you want to proceed with deletion? (y/N): ").lower() != 'y':
            logger.info("Deletion cancelled.")
            return
    
    strategy.execute()
    logger.info("Cache deletion complete!")

if __name__ == "__main__":
    main() 