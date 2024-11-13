import click
from pathlib import Path
from src.utils import setup_logger, Config
from src.data.preprocessing.tar_processor import TarProcessor

logger = setup_logger(__name__)

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--start-part', type=int, default=1,
              help='Starting part number')
@click.option('--end-part', type=int, default=200,
              help='Ending part number')
@click.option('--dataset', help='Override dataset name from config')
def main(config: str, start_part: int, end_part: int, dataset: str):
    """Download tar files from HuggingFace dataset."""
    config = Config(Path(config))
    
    # Use CLI override or config value
    dataset_name = dataset or config.get("dataset.name")
    
    processor = TarProcessor(dataset_name)
    processor.download_tar_range(start_part, end_part)

if __name__ == "__main__":
    main()
