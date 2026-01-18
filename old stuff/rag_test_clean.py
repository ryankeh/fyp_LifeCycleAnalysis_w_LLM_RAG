"""
LCA Product Embedding System with Qwen3-Embedding-0.6B
Handles CSV parsing, embedding generation, and efficient storage for 20k+ products

INSTALLATION:
pip install sentence-transformers pandas numpy h5py faiss-cpu
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import h5py
from typing import List, Dict, Optional, Tuple


def load_lca_dataset(filepath: str) -> pd.DataFrame:
    """
    Load LCA dataset from CSV/Excel file.
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        DataFrame with LCA product data
    """
    file_ext = Path(filepath).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(filepath)
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    print(f"Loaded {len(df)} products from {filepath}")
    return df


def create_sector_text(row: pd.Series, fields: List[str] = None) -> str:
    """
    Create searchable text representation from sector row.
    Combines sector name and industry descriptions.
    
    Args:
        row: DataFrame row
        fields: List of column names to include (default: sector-specific fields)
        
    Returns:
        Concatenated text string
    """
    if fields is None:
        # Default fields for sector data
        fields = [
            'Sector Name',
            'Industry sector descriptions'
        ]
    
    parts = []
    for field in fields:
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            parts.append(str(row[field]).strip())
    
    return " | ".join(parts)


def generate_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    text_fields: List[str] = None,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for all sectors in dataset.
    
    Args:
        df: DataFrame with sector data
        model: SentenceTransformer model
        text_fields: Fields to include in text representation (default: Sector Name + descriptions)
        batch_size: Batch size for encoding (larger = faster but more memory)
        show_progress: Show progress bar
        
    Returns:
        numpy array of embeddings (n_sectors, embedding_dim)
    """
    print("Creating text representations...")
    texts = [create_sector_text(row, text_fields) for _, row in df.iterrows()]
    
    print(f"Generating embeddings for {len(texts)} items...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def save_embeddings_hdf5(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    filepath: str,
    metadata: Dict = None,
    id_column: str = 'Sector Code',
    name_column: str = 'Sector Name'
):
    """
    Save embeddings and metadata to HDF5 file (efficient for large datasets).
    
    Args:
        embeddings: numpy array of embeddings
        df: Original DataFrame with sector data
        filepath: Output file path
        metadata: Additional metadata dict to save
        id_column: Column name for unique ID (default: 'Sector Code')
        name_column: Column name for display name (default: 'Sector Name')
    """
    with h5py.File(filepath, 'w') as f:
        # Save embeddings
        f.create_dataset('embeddings', data=embeddings, compression='gzip')
        
        # Save key identifiers (flexible for products or sectors)
        f.create_dataset('ids', data=df[id_column].astype(str).values)
        f.create_dataset('names', data=df[name_column].astype(str).values)
        
        # Save metadata
        if metadata:
            f.attrs['metadata'] = json.dumps(metadata)
        
        f.attrs['n_items'] = len(df)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['id_column'] = id_column
        f.attrs['name_column'] = name_column
    
    print(f"Saved embeddings to {filepath}")
    print(f"File size: {Path(filepath).stat().st_size / 1024 / 1024:.2f} MB")


def load_embeddings_hdf5(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load embeddings and metadata from HDF5 file.
    
    Returns:
        embeddings, ids, names, metadata
    """
    with h5py.File(filepath, 'r') as f:
        embeddings = f['embeddings'][:]
        ids = f['ids'][:].astype(str)
        names = f['names'][:].astype(str)
        
        metadata = json.loads(f.attrs.get('metadata', '{}'))
        
    print(f"Loaded {len(embeddings)} embeddings from {filepath}")
    return embeddings, ids, names, metadata


def save_embeddings_numpy(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    base_path: str,
    id_column: str = 'Sector Code',
    name_column: str = 'Sector Name'
):
    """
    Save embeddings and index using numpy files (simple alternative to HDF5).
    
    Args:
        embeddings: numpy array of embeddings
        df: Original DataFrame
        base_path: Base path without extension (e.g., 'embeddings/sectors')
        id_column: Column name for unique ID
        name_column: Column name for display name
    """
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    np.save(f"{base}_embeddings.npy", embeddings)
    
    # Save index mapping
    index_data = {
        'ids': df[id_column].tolist(),
        'names': df[name_column].tolist()
    }
    with open(f"{base}_index.json", 'w') as f:
        json.dump(index_data, f)
    
    print(f"Saved embeddings to {base}_embeddings.npy")


def load_embeddings_numpy(base_path: str) -> Tuple[np.ndarray, List, List]:
    """
    Load embeddings and index from numpy files.
    
    Returns:
        embeddings, ids, names
    """
    base = Path(base_path)
    
    embeddings = np.load(f"{base}_embeddings.npy")
    
    with open(f"{base}_index.json", 'r') as f:
        index_data = json.load(f)
    
    print(f"Loaded {len(embeddings)} embeddings from {base}_embeddings.npy")
    return embeddings, index_data['ids'], index_data['names']


def search_products(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Search for similar sectors given a query.
    
    Args:
        query: Search query string
        model: SentenceTransformer model
        embeddings: Pre-computed sector embeddings
        df: Original DataFrame
        top_k: Number of results to return
        
    Returns:
        DataFrame with top_k results and similarity scores
    """
    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Compute similarities
    similarities = model.similarity(query_embedding, embeddings)[0].numpy()
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Create results DataFrame
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]
    
    return results


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False):
    """
    Build FAISS index for fast similarity search (recommended for large datasets).
    
    Args:
        embeddings: numpy array of embeddings
        use_gpu: Use GPU acceleration if available
        
    Returns:
        FAISS index
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-cpu (or faiss-gpu)")
    
    embeddings_f32 = embeddings.astype('float32')
    dimension = embeddings_f32.shape[1]
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_f32)
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner product after normalization = cosine
    
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("Using GPU for FAISS")
    
    index.add(embeddings_f32)
    print(f"Built FAISS index with {index.ntotal} vectors")
    
    return index


def search_products_faiss(
    query: str,
    model: SentenceTransformer,
    faiss_index,
    df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Fast search using FAISS index.
    
    Args:
        query: Search query
        model: SentenceTransformer model
        faiss_index: Pre-built FAISS index
        df: Original DataFrame
        top_k: Number of results
        
    Returns:
        DataFrame with results and scores
    """
    import faiss
    
    # Encode and normalize query
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Create results
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = distances[0]
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    print("Loading Qwen3 model...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    
    # Load sector dataset
    print("\n" + "="*60)
    print("Processing Industry Sectors")
    print("="*60)
    df_sectors = load_lca_dataset("data\ceda_data.csv")
    
    # Generate embeddings for sectors
    embeddings_sectors = generate_embeddings(
        df_sectors,
        model,
        text_fields=None,  # Uses default: Sector Name + Industry sector descriptions
        batch_size=32
    )
    
    # Save sector embeddings
    save_embeddings_hdf5(
        embeddings_sectors,
        df_sectors,
        "sectors_embeddings.h5",
        metadata={"model": "Qwen/Qwen3-Embedding-0.6B", "type": "sectors"},
        id_column='Sector Code',
        name_column='Sector Name'
    )
    
    # Alternative: Save as numpy files
    # save_embeddings_numpy(embeddings_sectors, df_sectors, "embeddings/sectors")
    
    # Load embeddings (for next session)
    # embeddings, ids, names, metadata = load_embeddings_hdf5("sectors_embeddings.h5")
    
    # Search sectors
    query_sector = "renewable energy production"
    results = search_products(query_sector, model, embeddings_sectors, df_sectors, top_k=5)
    print(f"\nTop 5 sector results for: '{query_sector}'")
    print(results[['Sector Name', 'Industry sector descriptions', 'similarity_score']])
    
    # FAISS example (faster for large datasets)
    print("\n" + "="*60)
    print("FAISS Fast Search")
    print("="*60)
    faiss_index = build_faiss_index(embeddings_sectors)
    
    results_faiss = search_products_faiss(query_sector, model, faiss_index, df_sectors, top_k=5)
    print(f"\nFAISS results for: '{query_sector}'")
    print(results_faiss[['Sector Name', 'similarity_score']])