import torch
from pathlib import Path
from huggingface_hub import snapshot_download


def get_hf_model(model_id: str = "google/gemma-3-1b-it"):
    model_id = "google/gemma-3-1b-it"
    local_dir = Path.home() / "models" / model_id
    local_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(model_id, local_dir=local_dir)

    print("\nModel downloaded to:", model_path)
    return model_path


def detect_device(verbose: bool = True) -> str:
    if torch.cuda.is_available():
        dev = "cuda"
        if verbose:
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return dev

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
        if verbose:
            print("Using Apple MPS device")
        return dev

    if verbose:
        print("Using CPU")
    return "cpu"
