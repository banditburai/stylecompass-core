import os
import click
import pandas as pd
import pyarrow as pa
import lance
from pathlib import Path
import pickle
from tqdm import tqdm
from db_connection import init_db
import asyncio
import shutil
import numpy as np
import boto3
from src.utils import setup_logger, Config, init_db
from dotenv import load_dotenv

logger = setup_logger(__name__)

load_dotenv() 

def load_embeddings(embedding_paths):
    """Load embeddings and group by model type (v6/niji6)"""
    grouped_embeddings = {
        'v6': {},
        'niji6': {}
    }
    vector_dim = None
    
    for path in embedding_paths:
        path = Path(path)
        model_type = 'v6' if 'v6' in path.parts else 'niji6'
        sref_type = path.parts[-4]  # Gets the sref identifier
        
        embeddings_dict = {}
        logger.info(f"Loading embeddings from {path}")
        pkl_files = list(path.glob('*.pkl'))
        logger.info(f"Found {len(pkl_files)} pkl files")
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                embeddings = data['embeddings']
                filenames = data['filenames']
                
                if vector_dim is None and embeddings:
                    vector_dim = len(embeddings[0])
                elif embeddings and len(embeddings[0]) != vector_dim:
                    raise ValueError(f"Inconsistent vector dimensions: {len(embeddings[0])} vs {vector_dim}")
                
                for filename, emb in zip(filenames, embeddings):
                    embeddings_dict[filename] = emb
        
        if sref_type not in grouped_embeddings[model_type]:
            grouped_embeddings[model_type][sref_type] = []
        grouped_embeddings[model_type][sref_type].append(embeddings_dict)
    
    return grouped_embeddings, vector_dim

def calculate_mean_embeddings(grouped_embeddings):
    """Calculate mean embeddings for each model and sref combination"""
    mean_embeddings = {
        'v6': {},
        'niji6': {}
    }
    
    for model in grouped_embeddings:
        for sref in grouped_embeddings[model]:
            all_emb = np.stack([emb for emb_dict in grouped_embeddings[model][sref] 
                              for emb in emb_dict.values()])
            mean_embeddings[model][sref] = np.mean(all_emb, axis=0)
    
    return mean_embeddings

def upload_to_s3(image_path: Path, bucket_name: str) -> str:
    """Upload an image to S3 bucket in thumbnails/ prefix"""
    s3_client = boto3.client('s3',
        endpoint_url=os.getenv('AWS_ENDPOINT'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    key = f"thumbnails/{image_path.stem}.webp"
    
    try:
        s3_client.upload_file(str(image_path), os.getenv('AWS_BUCKET_NAME'), key)
        return f"{os.getenv('AWS_ENDPOINT')}/{os.getenv('AWS_BUCKET_NAME')}/{key}"
    except Exception as e:
        logger.error(f"Error uploading {image_path}: {str(e)}")
        raise

def process_images_and_embeddings(image_dir: Path, grouped_embeddings: dict, mean_embeddings: dict, vector_dim: int):
    """Process images and embeddings, keeping both individual and mean-pooled"""
    image_files = list(Path(image_dir).glob('*.webp'))
    logger.info(f"Found {len(image_files)} thumbnail images")
    
    records = []
    for img_path in tqdm(image_files, desc="Processing Images"):
        try:
            thumbnail_url = upload_to_s3(img_path, bucket_name=os.getenv('AWS_BUCKET_NAME'))
            img_name = img_path.stem
            
            v6_embs = grouped_embeddings['v6'].get(img_name, [])
            niji6_embs = grouped_embeddings['niji6'].get(img_name, [])
            v6_mean = mean_embeddings['v6'].get(img_name)
            niji6_mean = mean_embeddings['niji6'].get(img_name)
            
            if v6_embs and niji6_embs and v6_mean is not None and niji6_mean is not None:
                record = {
                    'image_name': img_name,
                    'thumbnail_url': thumbnail_url,
                    'v6_embeddings': v6_embs,
                    'niji6_embeddings': niji6_embs,
                    'v6_mean_embedding': v6_mean,
                    'niji6_mean_embedding': niji6_mean
                }
                records.append(record)
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
    
    logger.info(f"Created {len(records)} records out of {len(image_files)} images")
    return records

async def write_to_lance(image_dir: Path, embedding_paths: list):
    """Write to local Lance dataset first, then upload to cloud"""
    grouped_embeddings, vector_dim = load_embeddings(embedding_paths)
    logger.info(f"Vector dimension: {vector_dim}")
    
    if vector_dim is None:
        raise ValueError("No embeddings found to determine vector dimension")
    
    mean_embeddings = calculate_mean_embeddings(grouped_embeddings)
    records = process_images_and_embeddings(
        image_dir, 
        grouped_embeddings,
        mean_embeddings,
        vector_dim
    )
    
    schema = pa.schema([
        ('image_name', pa.string()),
        ('thumbnail_url', pa.string()),
        ('v6_embeddings', pa.list_(pa.list_(pa.float32(), vector_dim))),
        ('niji6_embeddings', pa.list_(pa.list_(pa.float32(), vector_dim))),
        ('v6_mean_embedding', pa.list_(pa.float32(), vector_dim)),
        ('niji6_mean_embedding', pa.list_(pa.float32(), vector_dim))
    ])

    table = pa.Table.from_pylist(records, schema=schema)
    reader = pa.RecordBatchReader.from_table(table)
    
    local_path = "temp_lance_dataset"
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        
    logger.info("Writing to local dataset...")
    lance.write_dataset(reader, local_path, schema)
    
    logger.info("Uploading to cloud...")
    db = await init_db()    
    
    table = await db.create_table(
        "sref_embeddings", 
        schema=schema, 
        mode="overwrite"
    )
    
    local_ds = lance.dataset(local_path)
    await table.add(local_ds)
    
    shutil.rmtree(local_path)
    logger.info("Upload complete and local files cleaned up")

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--batch-name', type=str, required=True,
              help='Name of the batch to process (e.g., batch_1_100)')
def main(config: str, batch_name: str):
    """Upload embeddings and thumbnails to Lance database"""
    config_path = Path(config)
    config_obj = Config(config_path)
    
    # Get paths from config
    batch_dir = Path(config_obj.get("paths.output_dir")) / batch_name
    thumbnail_dir = batch_dir / "thumbnails"
    embeddings_dir = batch_dir / "embeddings/csd_vit_large_sref_normal/1/database"
    
    if not thumbnail_dir.exists():
        raise click.BadParameter(f"Thumbnail directory not found at {thumbnail_dir}")
    if not embeddings_dir.exists():
        raise click.BadParameter(f"Embeddings directory not found at {embeddings_dir}")
    
    asyncio.run(write_to_lance(
        image_dir=thumbnail_dir,
        embedding_paths=[embeddings_dir]
    ))

if __name__ == '__main__':
    main()