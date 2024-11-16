# scripts/check_missing.py
import pandas as pd
from pathlib import Path
import yaml
import pickle
import json

def load_config():
    with open('configs/data_processing.yaml', 'r') as f:
        return yaml.safe_load(f)

def check_missing_embeddings(config):
    batch_start = config['batch']['current_start']
    batch_end = config['batch']['current_end']
    batch_name = f"batch_{batch_start}_{batch_end}"
    
    batch_dir = Path(config['paths']['output_dir']) / batch_name
    batch_csv = batch_dir / f"{batch_name}.csv"
    embeddings_dir = batch_dir / "embeddings/csd_vit_large_sref_normal/1/database"
    
    print(f"Checking batch CSV: {batch_csv}")
    print(f"Checking embeddings directory: {embeddings_dir}")
    
    batch_df = pd.read_csv(batch_csv)
    embeddings_files = [str(f) for f in embeddings_dir.glob("*.pkl")]
    
    database_embeddings = convert_embeddings(embeddings_files)
    
    # Create lookup from SREF ID to original paths
    sref_to_paths = {}
    for _, row in batch_df.iterrows():
        filename = str(Path(str(row['path'])).name)
        sref_id = filename.split('_')[0]
        if sref_id not in sref_to_paths:
            sref_to_paths[sref_id] = []
        sref_to_paths[sref_id].append(str(row['path']))
    
    # Group filenames by SREF ID and model
    sref_groups = {}
    for filename in database_embeddings['filenames']:
        base_name = filename.replace('.jpeg', '')
        sref_id = base_name.split('_')[0]
        if sref_id not in sref_groups:
            sref_groups[sref_id] = {'v6': [], 'niji6': []}
        
        if '_V6_' in base_name:
            sref_groups[sref_id]['v6'].append(base_name)
        elif '_NIJI6_' in base_name:
            sref_groups[sref_id]['niji6'].append(base_name)
    
    # Check for incomplete groups and collect missing info
    missing_info = {
        'v6': [],
        'niji6': []
    }
    
    for sref_id, group in sref_groups.items():
        if len(group['v6']) != 4:
            missing_info['v6'].append({
                'sref_id': sref_id,
                'found_count': len(group['v6']),
                'paths': sref_to_paths.get(sref_id, [])
            })
        if len(group['niji6']) != 4:
            missing_info['niji6'].append({
                'sref_id': sref_id,
                'found_count': len(group['niji6']),
                'paths': sref_to_paths.get(sref_id, [])
            })
    
    # Print summary
    for model in ['v6', 'niji6']:
        if missing_info[model]:
            print(f"\nIncomplete {model.upper()} groups:")
            for info in sorted(missing_info[model], key=lambda x: x['sref_id']):
                print(f"  SREF {info['sref_id']}: {info['found_count']}/4 images")
    
    return missing_info

def convert_embeddings(file_paths):
    database_embeddings = {'filenames': [], 'embeddings': []}
    for path in file_paths:
        with open(path, 'rb') as file:
            curr_embeddings = pickle.load(file)
        database_embeddings['filenames'] += curr_embeddings['filenames']
        database_embeddings['embeddings'] += curr_embeddings['embeddings']
    return database_embeddings

def main():
    config = load_config()
    missing_info = check_missing_embeddings(config)
    
    if any(missing_info.values()):
        batch_dir = Path(config['paths']['output_dir']) / f"batch_{config['batch']['current_start']}_{config['batch']['current_end']}"
        missing_file = batch_dir / 'missing_embeddings.json'
        
        # Save detailed missing info as JSON for easier reprocessing
        with open(missing_file, 'w') as f:
            json.dump({
                'batch_info': {
                    'start': config['batch']['current_start'],
                    'end': config['batch']['current_end']
                },
                'missing': missing_info
            }, f, indent=2)

if __name__ == "__main__":
    main()