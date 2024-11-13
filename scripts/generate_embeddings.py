import click
from pathlib import Path
from src.utils.config import Config
from src.models.factory import ModelFactory
from src.data.loaders import get_dataloaders, DataLoaderConfig
from src.models.extractors import FeatureExtractor

@click.command()
@click.option('--config', type=Path, required=True, help='Path to config file')
@click.option('--data-dir', type=Path, required=True, help='Dataset directory')
@click.option('--skip-val', is_flag=True, help='Skip validation set processing')
def main(config: Path, data_dir: Path, skip_val: bool):
    """Generate embeddings for image dataset"""
    
    # Load config
    config = Config(config)
    
    # Create model and transform
    model_factory = ModelFactory()
    model, transform = model_factory.create_model(config.model)
    model.eval()
    
    # Setup dataloaders with correct batch size
    loader_config = DataLoaderConfig(
        batch_size=32 if config.features.type == 'gram' else config.training.batch_size,
        num_workers=config.data.num_workers,
        distributed=config.training.distributed.enabled,
        world_size=config.training.distributed.world_size,
        feature_type=config.features.type,
        pin_memory=True,
        drop_last=False
    )
    
    query_loader, values_loader = get_dataloaders(
        dataset_name=config.data.dataset,
        data_dir=data_dir,
        transform=transform,
        config=loader_config
    )
    
    print(f"train: {len(values_loader.dataset)} imgs / query: {len(query_loader.dataset)} imgs")

    # Create embedding save path
    embed_dir = Path(config.output.embed_dir)
    embsavepath = (
        embed_dir / 
        f'{config.model.pt_style}_{config.model.arch}_{config.data.dataset}_{config.features.type}' /
        str(config.features.layer)
    )
    
    if config.features.type == 'gram':
        path1, path2 = str(embsavepath).split('_gram')
        embsavepath = Path('_'.join([
            path1, 'gram', str(config.features.gram_dims), 
            config.features.qsplit, path2
        ]))
    
    # Check if validation set exists
    valexist = (embsavepath / 'database/embeddings_0.pkl').exists() or skip_val
    
    # Extract features based on model type
    extractor = FeatureExtractor(config.features)
    
    # Extract query features
    query_features = extractor.extract_features(
        model, 
        query_loader,
        use_cuda=True,
        use_fp16=config.training.use_fp16
    )
    
    # Save query features
    extractor.save_features(
        query_features,
        query_loader.dataset.namelist,
        embsavepath,
        config.features.qsplit
    )
    
    # Extract and save validation features if needed
    if not valexist:
        values_features = extractor.extract_features(
            model, 
            values_loader,
            use_cuda=True,
            use_fp16=config.training.use_fp16
        )
        extractor.save_features(
            values_features,
            values_loader.dataset.namelist,
            embsavepath,
            'database'
        )
    
    print(f'Embeddings saved to: {embsavepath}')

if __name__ == '__main__':
    main()