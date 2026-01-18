"""
Example usage of Qwen3-Embedding-0.6B for product similarity search
Suitable for LCA/GHG emissions database matching

INSTALLATION:
pip install sentence-transformers
pip install faiss-cpu  # optional, for large-scale search
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the official Qwen3 embedding model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Example: LCA product database
products_db = [
    "Steel rebar, hot-rolled, virgin material",
    "Aluminum sheet, recycled content 50%",
    "Concrete, ready-mix, Portland cement CEM I",
    "Polyethylene terephthalate (PET), virgin granulate",
    "Glass fiber reinforced polymer (GFRP)",
    "Steel plate, cold-rolled, galvanized",
    "Aluminum extrusion, virgin material",
    "Concrete block, lightweight aggregate"
]

# Embed the entire product database
print("Embedding product database...")
db_embeddings = model.encode(products_db)
print(f"Database embeddings shape: {db_embeddings.shape}")

# Query: User wants to find similar product
query = "galvanized steel sheet"

# Embed the query
print(f"\nQuery: '{query}'")
query_embedding = model.encode([query])

# Compute similarity using built-in method
similarities = model.similarity(query_embedding, db_embeddings)[0]
print(f"Similarities shape: {similarities.shape}")

# Rank results
ranked_indices = np.argsort(similarities.numpy())[::-1]

# Display top 3 matches
print("\nTop 3 matches:")
for i, idx in enumerate(ranked_indices[:3], 1):
    print(f"{i}. {products_db[idx]}")
    print(f"   Similarity: {similarities[idx]:.4f}\n")

# Batch processing example
print("="*60)
print("BATCH QUERY EXAMPLE")
print("="*60)

queries = [
    "recycled aluminum",
    "lightweight concrete",
    "virgin PET plastic"
]

# Embed all queries at once (more efficient)
query_embeddings = model.encode(queries)

# Compute all similarities at once
all_similarities = model.similarity(query_embeddings, db_embeddings)

# Find best match for each query
for i, q in enumerate(queries):
    sims = all_similarities[i]
    best_idx = np.argmax(sims.numpy())
    print(f"\nQuery: '{q}'")
    print(f"Best match: {products_db[best_idx]} (score: {sims[best_idx]:.4f})")

# Cross-similarity example (find similar products within database)
print("\n" + "="*60)
print("FIND SIMILAR PRODUCTS IN DATABASE")
print("="*60)

# Compute similarity between all products
product_similarities = model.similarity(db_embeddings, db_embeddings)

# For first product, find most similar (excluding itself)
target_idx = 0
print(f"\nProducts similar to: '{products_db[target_idx]}'")
sims = product_similarities[target_idx].numpy()
# Exclude the product itself by setting its similarity to -1
sims[target_idx] = -1
top_similar = np.argsort(sims)[::-1][:3]

for i, idx in enumerate(top_similar, 1):
    print(f"{i}. {products_db[idx]} (similarity: {sims[idx]:.4f})")

# FAISS example for large-scale search
print("\n" + "="*60)
print("FAISS EXAMPLE (for large databases)")
print("="*60)

try:
    import faiss
    
    # Convert to numpy float32 for FAISS
    db_embeddings_np = db_embeddings.numpy().astype('float32')
    
    # Create FAISS index
    dimension = db_embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(db_embeddings_np)
    index.add(db_embeddings_np)
    
    # Query
    query_emb_np = query_embedding.numpy().astype('float32')
    faiss.normalize_L2(query_emb_np)
    
    # Search for top-k similar items
    k = 3
    distances, indices = index.search(query_emb_np, k)
    
    print(f"\nFAISS Top {k} results for '{query}':")
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        print(f"{i}. {products_db[idx]} (similarity: {dist:.4f})")
        
except ImportError:
    print("\nFAISS not installed. Install with: pip install faiss-cpu")
    print("For GPU support: pip install faiss-gpu")

# Production tip: Save/load embeddings to avoid re-computing
print("\n" + "="*60)
print("SAVE/LOAD EMBEDDINGS")
print("="*60)

# Save embeddings
np.save('product_embeddings.npy', db_embeddings.numpy())
print("Embeddings saved to 'product_embeddings.npy'")

# Load embeddings (much faster than re-encoding)
# loaded_embeddings = np.load('product_embeddings.npy')
# print(f"Loaded embeddings shape: {loaded_embeddings.shape}")