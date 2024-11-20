from pathlib import Path
from typing import Dict, List, Tuple, TypedDict, Optional
import asyncio
import pickle
from dataclasses import dataclass

import click
import numpy as np
import pyarrow as pa
import lance
from tqdm import tqdm

from src.utils import setup_logger, Config, init_db

# Constants
DEFAULT_CONFIG = Path('configs/data_processing.yaml')
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / "upsert_embeddings.log"
logger = setup_logger(__name__, log_file=log_file)


class EmbeddingData(TypedDict):
    embeddings: List[np.ndarray]
    filenames: List[str]


@dataclass
class GroupedEmbeddings:
    v6: Dict[str, np.ndarray]
    niji6: Dict[str, np.ndarray]
    vector_dim: Optional[int] = None


def load_embeddings(embedding_paths: List[Path]) -> GroupedEmbeddings:
    """Load embeddings and group by model type (v6/niji6)."""
    grouped = GroupedEmbeddings(v6={}, niji6={})
    
    for path in embedding_paths:
        logger.info(f"Loading embeddings from {path}")
        pkl_files = list(path.glob('*.pkl'))
        logger.info(f"Found {len(pkl_files)} pkl files")
        
        for pkl_file in pkl_files:
            logger.info(f"Loading {pkl_file}")
            data: EmbeddingData = pickle.loads(pkl_file.read_bytes())
            
            if not grouped.vector_dim and data['embeddings']:
                grouped.vector_dim = len(data['embeddings'][0])
            
            for filename, emb in zip(data['filenames'], data['embeddings']):
                base_name = Path(filename).stem
                if '_V6_' in base_name:
                    grouped.v6[base_name] = emb
                elif '_NIJI6_' in base_name:
                    grouped.niji6[base_name] = emb
            
            logger.info(f"Processed batch with {len(data['embeddings'])} embeddings")
    
    logger.info(f"Total v6 embeddings: {len(grouped.v6)}")
    logger.info(f"Total niji6 embeddings: {len(grouped.niji6)}")
    logger.info("Sample V6 keys:", list(grouped.v6.keys())[:5])
    
    return grouped


def get_schemas(vector_dim: int) -> Tuple[pa.Schema, pa.Schema]:
    """Create schemas for individual and mean embeddings."""
    individual_schema = pa.schema([
        ('image_name', pa.string()),
        ('thumbnail_name', pa.string()),
        ('embedding', pa.list_(pa.float32(), vector_dim)),
        ('model_type', pa.string())
    ])
    
    mean_schema = pa.schema([
        ('sref_id', pa.string()),
        ('thumbnail_urls', pa.list_(pa.string())),
        ('embedding', pa.list_(pa.float32(), vector_dim))
    ])
    
    return individual_schema, mean_schema


async def upsert_to_both_dbs(
    local_db: lance.LanceDb,
    remote_db: lance.LanceDb,
    table_name: str,
    records: List[dict],
    schema: pa.Schema
) -> None:
    """Write records to both local and remote Lance DBs."""
    # Local DB
    if table_name in local_db.table_names():
        local_table = local_db.open_table(table_name)
    else:
        local_table = local_db.create_table(table_name, schema=schema)
    local_table.add(pa.Table.from_pylist(records, schema=schema))
    
    # Remote DB
    if table_name in await remote_db.table_names():
        remote_table = await remote_db.open_table(table_name)
    else:
        remote_table = await remote_db.create_table(table_name, schema=schema)
    await remote_table.add(pa.Table.from_pylist(records, schema=schema))


async def write_to_lance(image_dir: Path, embedding_paths: List[Path]) -> None:
    """Write embeddings to both local and remote Lance datasets."""
    grouped = load_embeddings(embedding_paths)
    if not grouped.vector_dim:
        raise ValueError("No embeddings found to determine vector dimension")
    
    individual_schema, mean_schema = get_schemas(grouped.vector_dim)
    
    # Process images
    image_files = list(Path(image_dir).glob('*.webp'))
    logger.info(f"Found {len(image_files)} thumbnail images")
    
    thumbnail_names = {img_path.stem: f"{img_path.stem}.webp" for img_path in image_files}
    
    # Initialize records and groups
    individual_records = []
    sref_groups: Dict[str, Dict[str, List[Tuple[str, str, np.ndarray]]]] = {}
    
    # Process individual images
    for img_path in tqdm(image_files, desc="Processing individual images"):
        image_name = img_path.stem
        thumbnail_name = thumbnail_names.get(image_name)
        
        if not thumbnail_name:
            logger.warning(f"Missing thumbnail for {image_name}")
            continue
            
        # Group by SREF ID
        sref_id = image_name.split('_')[0]
        if sref_id not in sref_groups:
            sref_groups[sref_id] = {'v6': [], 'niji6': []}
        
        if '_V6_' in image_name:
            emb = grouped.v6.get(image_name)
            model_type = 'v6'
            if emb is not None:
                sref_groups[sref_id]['v6'].append((image_name, thumbnail_name, emb))
        else:
            emb = grouped.niji6.get(image_name)
            model_type = 'niji6'
            if emb is not None:
                sref_groups[sref_id]['niji6'].append((image_name, thumbnail_name, emb))
        
        if emb is not None:
            individual_records.append({
                'image_name': image_name,
                'thumbnail_name': thumbnail_name,
                'embedding': emb,
                'model_type': model_type
            })
    
    # Process mean embeddings
    v6_mean_records = []
    niji6_mean_records = []
    missing = {'v6': 0, 'niji6': 0}
    
    for sref_id, group in tqdm(sref_groups.items(), desc="Processing mean embeddings"):
        for model_type, records in [('v6', v6_mean_records), ('niji6', niji6_mean_records)]:
            if len(group[model_type]) == 4:
                embeddings = [emb for _, _, emb in sorted(group[model_type])]
                urls = [url for _, url, _ in sorted(group[model_type])]
                records.append({
                    'sref_id': sref_id,
                    'thumbnail_urls': urls,
                    'embedding': np.mean(embeddings, axis=0)
                })
            else:
                missing[model_type] += 1
    
    # Initialize DBs and write records
    local_db = lance.LanceDb(path="data/lance_db")
    remote_db = await init_db()
    
    await upsert_to_both_dbs(local_db, remote_db, "sref_individual_embeddings", 
                            individual_records, individual_schema)
    await upsert_to_both_dbs(local_db, remote_db, "sref_v6_mean_embeddings", 
                            v6_mean_records, mean_schema)
    await upsert_to_both_dbs(local_db, remote_db, "sref_niji6_mean_embeddings", 
                            niji6_mean_records, mean_schema)
    
    logger.info(f"Created {len(individual_records)} individual embedding records")
    logger.info(f"Created {len(v6_mean_records)} V6 mean records")
    logger.info(f"Created {len(niji6_mean_records)} NIJI6 mean records")
    logger.info(f"Missing V6 embeddings: {missing['v6']}")
    logger.info(f"Missing NIJI6 embeddings: {missing['niji6']}")


@click.command()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_CONFIG,
    help='Path to config file'
)
@click.option(
    '--batch-name',
    type=str,
    help='Name of batch to process (e.g., batch_1_100). Defaults to current batch from config.'
)
def main(config: Path, batch_name: Optional[str]) -> None:
    """Upload embeddings to Lance database."""
    config_obj = Config(config)
    
    if batch_name is None:
        start = config_obj.get("batch.current_start")
        end = config_obj.get("batch.current_end")
        batch_name = f"batch_{start}_{end}"
        logger.info(f"Using current batch from config: {batch_name}")
    
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