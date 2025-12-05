from pathlib import Path

from huggingface_hub import snapshot_download
from inferlite.weight_loader import load_all_tensors


def main():

    model_id = "google/gemma-3-1b-it"
    local_dir = Path.home() / "models" / model_id
    local_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(model_id, local_dir=local_dir)

    print("\nModel downloaded to:", model_path)

    tensors = load_all_tensors(model_path, device="cpu")


if __name__ == "__main__":
    main()
