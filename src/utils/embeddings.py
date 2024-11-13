# src/utils/embeddings.py
import os
import pickle
import math
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np

def save_chunk(
    embeddings: List[np.ndarray],
    filenames: List[str],
    count: int,
    chunk_dir: Union[str, Path],
    chunk_size: int = 50000
) -> None:
    """Saves embeddings and filenames in chunks"""
    chunk_dir = Path(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    if len(embeddings) < chunk_size:
        data = {
            'embeddings': embeddings,
            'filenames': filenames,
        }
        with open(chunk_dir / f'embeddings_{count}.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        for i in range(0, math.ceil(len(embeddings)/chunk_size)):
            data = {
                'embeddings': embeddings[i*chunk_size: min((i+1)*chunk_size, len(embeddings))],
                'filenames': filenames[i*chunk_size: min((i+1)*chunk_size, len(embeddings))],
            }
            with open(chunk_dir / f'embeddings_{i}.pkl', 'wb') as f:
                pickle.dump(data, f)

class EmbeddingIO:
    """Handles reading and writing of embeddings"""
    
    @staticmethod
    def load_chunk(filename: Union[str, Path]) -> Dict[str, Any]:
        """Load a chunk of embeddings from disk"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
            
    @staticmethod
    def save_embeddings(
        embeddings: np.ndarray,
        filenames: List[str],
        save_dir: Path,
        split: str = 'database'
    ) -> None:
        """Save embeddings to disk in chunks"""
        embeddings_np = np.asarray(embeddings.cpu().detach(), dtype=np.float16)
        embeddings_list = list(embeddings_np)
        
        save_dir = Path(save_dir) / split
        save_chunk(embeddings_list, filenames, 0, save_dir)