import click
from pathlib import Path
import sys
from src.utils import setup_logger, Config, BatchHelper
from typing import Optional

# Add stylepriors to path
STYLEPRIORS_PATH = Path(__file__).parent.parent / "stylepriors"
sys.path.append(str(STYLEPRIORS_PATH))

logger = setup_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--next-batch', is_flag=True, help='Use next batch range')
@click.option('--batch-size', '-b', type=int, default=32,
              help='Batch size for embedding generation')
@click.option('--num-workers', '-j', type=int, default=8,
              help='Number of data loading workers')
@click.option('--checkpoint', type=click.Path(exists=True),
              default='models/checkpoint.pth',
              help='Path to model checkpoint')
def main(config: str, next_batch: bool, batch_size: int, num_workers: int, checkpoint: str):
    """Generate style embeddings for a batch of processed images."""
    config_path = Path(config)
    config_obj = Config(config_path)
    batch_helper = BatchHelper(config_path)
    
    if next_batch:
        start_part, end_part = batch_helper.next_batch()
    else:
        start_part, end_part = batch_helper.get_current_batch()
    
    # Get paths from config
    batch_name = f"batch_{start_part}_{end_part}"
    batch_dir = Path(config_obj.get("paths.output_dir")) / batch_name
    dataset_dir = batch_dir / "dataset_temp"  # Add this line for the inner folder
    embeddings_dir = batch_dir / "embeddings"  # Change this to save in batch folder
    
    # Ensure batch has been processed
    csv_path = batch_dir / f"{batch_name}.csv"
    if not csv_path.exists():
        logger.error(f"Batch CSV not found at {csv_path}. Run prepare-batch first.")
        return
        
    logger.info(f"Generating embeddings for batch {start_part} to {end_part}")
    logger.info(f"Using batch size {batch_size} and {num_workers} workers")
    
    # Import main_sim here to avoid early import issues
    from stylepriors.main_sim import main as stylepriors_main
    import sys
    
    # Prepare arguments for stylepriors
    sys.argv = [
        "prepare_embeddings",
        "--dataset", "sref",
        "-a", "vit_large",
        "--pt_style", "csd",
        "--feattype", "normal",
        "--world-size", "1",
        "--dist-url", "tcp://localhost:6001",
        "-b", str(batch_size),
        "-j", str(num_workers),
        "--embed_dir", str(embeddings_dir),
        "--data-dir", str(dataset_dir),
        "--model_path", str(checkpoint),
        "--csv-path", str(csv_path)
    ]
    
    try:
        stylepriors_main()
        logger.info(f"Successfully generated embeddings for batch {batch_name}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

if __name__ == '__main__':
    main()