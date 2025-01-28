from sentence_transformers import SentenceTransformer
import os

def download_model():
    """Download the model files to local directory."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    models_dir = os.getenv('MODELS_DIR', 'storage/models')
    local_model_path = os.path.join(models_dir, 'all-MiniLM-L6-v2')
    
    # Create directory if it doesn't exist
    os.makedirs(local_model_path, exist_ok=True)
    
    print(f"Downloading model {model_name}...")
    try:
        # Download and save the model
        model = SentenceTransformer(model_name, cache_folder=models_dir)
        model.save(local_model_path)
        print(f"[SUCCESS] Model saved to {local_model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        raise

if __name__ == "__main__":
    download_model()