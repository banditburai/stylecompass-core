import os
import click
import pandas as pd
import pyarrow as pa
import lance
from pathlib import Path
import pickle
from tqdm import tqdm
import asyncio
import shutil
import numpy as np
import boto3
from boto3.session import Config as BotoConfig
from src.utils import setup_logger, Config, init_db
from dotenv import load_dotenv
from typing import cast
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
from tenacity import retry, stop_after_attempt, wait_exponential

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "upsert_embeddings.log"
logger = setup_logger(__name__, log_file=log_file)

load_dotenv()

BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_ENDPOINT = os.getenv('AWS_ENDPOINT')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

if not all([BUCKET_NAME, AWS_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
    raise ValueError("Missing required AWS configuration. Please check your .env file.")

BUCKET_NAME = cast(str, BUCKET_NAME)
AWS_ENDPOINT = cast(str, AWS_ENDPOINT)
AWS_ACCESS_KEY_ID = cast(str, AWS_ACCESS_KEY_ID)
AWS_SECRET_ACCESS_KEY = cast(str, AWS_SECRET_ACCESS_KEY)

def load_embeddings(embedding_paths):
    """Load embeddings and group by model type (v6/niji6)"""
    grouped_embeddings = {
        'v6': {},
        'niji6': {}
    }
    vector_dim = None
    
    for path in embedding_paths:
        path = Path(path)
        logger.info(f"Loading embeddings from {path}")
        pkl_files = list(path.glob('*.pkl'))
        logger.info(f"Found {len(pkl_files)} pkl files")
        
        for pkl_file in pkl_files:
            logger.info(f"Loading {pkl_file}")
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                embeddings = data['embeddings']
                filenames = data['filenames']
                logger.info(f"Loaded {len(filenames)} embeddings from {pkl_file}")
                
                if vector_dim is None and embeddings:
                    vector_dim = len(embeddings[0])
                
                for filename, emb in zip(filenames, embeddings):
                    # Remove extension and convert to base name
                    base_name = Path(filename).stem  # This removes .png
                    
                    # Determine model type from filename
                    if '_V6_' in base_name:
                        grouped_embeddings['v6'][base_name] = emb
                    elif '_NIJI6_' in base_name:
                        grouped_embeddings['niji6'][base_name] = emb
                
                logger.info(f"Processed batch with {len(embeddings)} embeddings")
    
    # Add some debug logging
    logger.info(f"Total v6 embeddings: {len(grouped_embeddings['v6'])}")
    logger.info(f"Total niji6 embeddings: {len(grouped_embeddings['niji6'])}")
    
    # Print a few example keys to verify matching
    logger.info("Sample V6 keys with embeddings:")
    for key in list(grouped_embeddings['v6'].keys())[:5]:
        logger.info(f"  {key}")
    
    return grouped_embeddings, vector_dim

def calculate_mean_embeddings(grouped_embeddings: dict) -> dict:
    """Calculate mean embeddings for each image"""
    mean_embeddings = {
        'v6': {},
        'niji6': {}
    }
    
    # For each model type (v6/niji6)
    for model in ['v6', 'niji6']:
        # For each image
        for img_name, embedding in grouped_embeddings[model].items():
            # No need to calculate mean since we're storing individual embeddings now
            mean_embeddings[model][img_name] = embedding
    
    logger.info(f"Calculated mean embeddings for {len(mean_embeddings['v6'])} V6 images")
    logger.info(f"Calculated mean embeddings for {len(mean_embeddings['niji6'])} NIJI6 images")
    
    return mean_embeddings

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True
)
def upload_to_s3(image_path: Path, bucket_name: str) -> str:
    """Upload an image to S3 bucket in thumbnails/ prefix with exponential backoff"""
    s3_client = boto3.client('s3',
        endpoint_url=AWS_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-east-1',
        config=BotoConfig(
            retries=dict(
                max_attempts=3,
                mode='adaptive'
            ),
            connect_timeout=10,
            read_timeout=20
        )
    )
    key = f"thumbnails/{image_path.stem}.webp"
    
    try:
        s3_client.upload_file(str(image_path), BUCKET_NAME, key)
        return f"{AWS_ENDPOINT}/{BUCKET_NAME}/{key}"
    except Exception as e:
        logger.error(f"Error uploading {image_path}: {str(e)}")
        raise

def upload_with_retry(img_path: Path, bucket_name: str, max_retries: int = 3) -> tuple[str, str]:
    """Upload a single image with retry logic and backoff"""
    for attempt in range(max_retries):
        try:
            thumbnail_url = upload_to_s3(img_path, bucket_name=bucket_name)
            return img_path.stem, thumbnail_url
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to upload {img_path} after {max_retries} attempts: {e}")
                raise
            logger.warning(f"Retry {attempt + 1}/{max_retries} for {img_path}: {e}")
            # Add exponential backoff between retries
            time.sleep(min(30, (2 ** attempt)))  # 1s, 2s, 4s...
    raise Exception("Should not reach here")

async def write_to_lance(image_dir: Path, embedding_paths: list):
    """Write to local Lance dataset first, then upload to cloud"""
    grouped_embeddings, vector_dim = load_embeddings(embedding_paths)
    logger.info(f"Vector dimension: {vector_dim}")
    
    if vector_dim is None:
        raise ValueError("No embeddings found to determine vector dimension")
    
    # Define schemas
    individual_schema = pa.schema([
        ('image_name', pa.string()),
        ('thumbnail_url', pa.string()),
        ('embedding', pa.list_(pa.float32(), vector_dim)),
        ('model_type', pa.string())
    ])
    
    mean_schema = pa.schema([
        ('sref_id', pa.string()),
        ('thumbnail_urls', pa.list_(pa.string())),
        ('embedding', pa.list_(pa.float32(), vector_dim))
    ])
    
    # First upload all thumbnails and collect URLs
    image_files = list(Path(image_dir).glob('*.webp'))
    logger.info(f"Found {len(image_files)} thumbnail images")
    
    upload_func = partial(upload_with_retry, bucket_name=BUCKET_NAME)
    uploaded_urls = {}
    
    # Upload in batches of 100
    batch_size = 100
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i in range(0, len(image_files), batch_size):
            batch_number = i // batch_size
            
            # Extra sleep every 10th batch
            if batch_number > 0 and batch_number % 10 == 0:
                logger.info(f"Taking extra rest before batch {batch_number}")
                time.sleep(3)
            
            batch = image_files[i:i + batch_size]
            futures = [executor.submit(upload_func, img_path) for img_path in batch]
            
            for future in tqdm(futures, total=len(batch), 
                             desc=f"Uploading batch {batch_number + 1}/{len(image_files)//batch_size + 1}"):
                try:
                    img_name, thumbnail_url = future.result()
                    uploaded_urls[img_name] = thumbnail_url
                except Exception as e:
                    logger.error(f"Error uploading image: {e}")
            
            time.sleep(1.5)  # Regular sleep between batches
    
    # Now process embeddings and create records
    individual_records = []
    sref_groups = {}
    
    # First create individual records and group by SREF
    for img_path in tqdm(image_files, desc="Processing individual images"):
        image_name = img_path.stem
        thumbnail_url = uploaded_urls.get(image_name)
        
        if thumbnail_url is None:
            logger.warning(f"Missing thumbnail URL for {image_name}")
            continue
            
        # Group by SREF ID
        sref_id = image_name.split('_')[0]
        if sref_id not in sref_groups:
            sref_groups[sref_id] = {'v6': [], 'niji6': []}
        
        if '_V6_' in image_name:
            emb = grouped_embeddings['v6'].get(image_name)
            model_type = 'v6'
            if emb is not None:
                sref_groups[sref_id]['v6'].append((image_name, thumbnail_url, emb))
        else:
            emb = grouped_embeddings['niji6'].get(image_name)
            model_type = 'niji6'
            if emb is not None:
                sref_groups[sref_id]['niji6'].append((image_name, thumbnail_url, emb))
        
        if emb is not None:
            individual_records.append({
                'image_name': image_name,
                'thumbnail_url': thumbnail_url,
                'embedding': emb,
                'model_type': model_type
            })
    
    # Create mean-pooled records
    v6_mean_records = []
    niji6_mean_records = []
    missing_v6 = 0
    missing_niji6 = 0
    
    for sref_id, group in tqdm(sref_groups.items(), desc="Processing mean embeddings"):
        # Process V6
        if len(group['v6']) == 4:
            v6_embeddings = [emb for _, _, emb in sorted(group['v6'])]
            v6_urls = [url for _, url, _ in sorted(group['v6'])]
            v6_mean_records.append({
                'sref_id': sref_id,
                'thumbnail_urls': v6_urls,
                'embedding': np.mean(v6_embeddings, axis=0)
            })
        else:
            missing_v6 += 1
        
        # Process NIJI6
        if len(group['niji6']) == 4:
            niji6_embeddings = [emb for _, _, emb in sorted(group['niji6'])]
            niji6_urls = [url for _, url, _ in sorted(group['niji6'])]
            niji6_mean_records.append({
                'sref_id': sref_id,
                'thumbnail_urls': niji6_urls,
                'embedding': np.mean(niji6_embeddings, axis=0)
            })
        else:
            missing_niji6 += 1
    
    # Write to LanceDB
    db = await init_db()
    
    # Write individual embeddings
    individual_table = await db.create_table(
        "sref_individual_embeddings",
        schema=individual_schema,
        mode="overwrite"
    )
    await individual_table.add(pa.Table.from_pylist(individual_records, schema=individual_schema))
    
    # Write V6 mean embeddings
    v6_mean_table = await db.create_table(
        "sref_v6_mean_embeddings",
        schema=mean_schema,
        mode="overwrite"
    )
    await v6_mean_table.add(pa.Table.from_pylist(v6_mean_records, schema=mean_schema))
    
    # Write NIJI6 mean embeddings
    niji6_mean_table = await db.create_table(
        "sref_niji6_mean_embeddings",
        schema=mean_schema,
        mode="overwrite"
    )
    await niji6_mean_table.add(pa.Table.from_pylist(niji6_mean_records, schema=mean_schema))
    
    logger.info(f"Created {len(individual_records)} individual embedding records")
    logger.info(f"Created {len(v6_mean_records)} V6 mean records")
    logger.info(f"Created {len(niji6_mean_records)} NIJI6 mean records")
    logger.info(f"Missing V6 embeddings: {missing_v6}")
    logger.info(f"Missing NIJI6 embeddings: {missing_niji6}")

@click.command()
@click.option('--config', type=click.Path(exists=True), 
              default='configs/data_processing.yaml',
              help='Path to config file')
@click.option('--batch-name', type=str, required=False,
              help='Name of the batch to process (e.g., batch_1_100). Defaults to current batch from config.')
def main(config: str, batch_name: str | None):
    """Upload embeddings and thumbnails to Lance database"""
    config_path = Path(config)
    config_obj = Config(config_path)
    
    if batch_name is None:
        # Get current batch from config
        start = config_obj.get("batch.current_start")
        end = config_obj.get("batch.current_end")
        batch_name = f"batch_{start}_{end}"
        logger.info(f"Using current batch from config: {batch_name}")
    
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