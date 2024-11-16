import click
import sys
from pathlib import Path
from src.utils import setup_logger, Config, BatchHelper
from typing import Optional
import subprocess

script_name = Path(__file__).stem
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"{script_name}.log"  # where script_name matches the script's purpose
logger = setup_logger(__name__, log_file=log_file)

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
              default=None, 
              help='Path to model checkpoint')
def main(config: str, next_batch: bool, batch_size: int, num_workers: int, checkpoint: Optional[str]):
    config_path = Path(config)
    config_obj = Config(config_path)
    
    # Get stylepriors path from config
    stylepriors_path = Path(config_obj.get("paths.stylepriors_path"))
    if not stylepriors_path.exists():
        raise click.BadParameter(f"Stylepriors path not found at {stylepriors_path}")
    
    # Use checkpoint from config if not specified
    checkpoint_path = Path(checkpoint) if checkpoint else Path(config_obj.get("paths.checkpoint_path"))
    if not checkpoint_path.exists():
        raise click.BadParameter(f"Checkpoint not found at {checkpoint_path}")
    
    batch_helper = BatchHelper(config_path)
    
    if next_batch:
        start_part, end_part = batch_helper.next_batch()
    else:
        start_part, end_part = batch_helper.get_current_batch()
    
    # Get paths from config
    batch_name = f"batch_{start_part}_{end_part}"
    batch_dir = Path(config_obj.get("paths.output_dir")) / batch_name
    dataset_dir = batch_dir / "dataset_temp"
    embeddings_dir = batch_dir / "embeddings"
    
    # Ensure batch has been processed
    csv_path = batch_dir / f"{batch_name}.csv"
    if not csv_path.exists():
        logger.error(f"Batch CSV not found at {csv_path}. Run prepare-batch first.")
        return
        
    logger.info(f"Generating embeddings for batch {start_part} to {end_part}")
    logger.info(f"Using batch size {batch_size} and {num_workers} workers")
    
    # Build command
    cmd = [
        "python",
        str(stylepriors_path / "main_sim.py"),
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
        "--model_path", str(checkpoint_path),
        "--csv-path", str(csv_path)
    ]
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, text=True)
        logger.info(f"Successfully generated embeddings for batch {batch_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

if __name__ == '__main__':
    main()