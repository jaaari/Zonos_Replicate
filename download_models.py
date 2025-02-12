import os
from zonos.model import Zonos

def download_models():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download both versions
    models = {
        "transformer": "Zyphra/Zonos-v0.1-transformer",
        "hybrid": "Zyphra/Zonos-v0.1-hybrid"
    }
    
    for model_type, model_path in models.items():
        print(f"Downloading {model_type} model...")
        model = Zonos.from_pretrained(model_path)
        save_path = f"models/{model_type}"
        model.save_pretrained(save_path)
        print(f"Saved {model_type} model to {save_path}")

if __name__ == "__main__":
    download_models() 