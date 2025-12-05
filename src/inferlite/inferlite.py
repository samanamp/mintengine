import os
from pathlib import Path
from safetensors import safe_open
from huggingface_hub import snapshot_download

def analyze_weight_sizes(tensors):
    stats = {
        "embed": 0,
        "final_norm": 0,
        "layers": {},
        "mlp_total": 0,
        "attn_total": 0,
        "layernorm_total": 0,
        "total": 0,
    }

    for name, t in tensors.items():
        size = t.numel() * t.element_size()
        stats["total"] += size

        # embedding layer
        if name.startswith("model.embed_tokens"):
            stats["embed"] += size
            continue

        # final norm
        if name == "model.norm.weight":
            stats["final_norm"] += size
            continue

        # layer index extraction
        if name.startswith("model.layers."):
            after = name[len("model.layers.") :]
            idx = int(after.split(".", 1)[0])

            if idx not in stats["layers"]:
                stats["layers"][idx] = {
                    "total": 0,
                    "mlp": 0,
                    "attn": 0,
                    "layernorm": 0,
                }

            layer = stats["layers"][idx]
            layer["total"] += size

            # classify by component
            if ".mlp." in name:
                layer["mlp"] += size
                stats["mlp_total"] += size
            elif ".self_attn." in name:
                layer["attn"] += size
                stats["attn_total"] += size
            elif "layernorm" in name:
                layer["layernorm"] += size
                stats["layernorm_total"] += size

    return stats

# ANSI colors
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

def format_size(num_bytes):
    """Return a human friendly size with automatic units."""
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GiB"
    elif num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.2f} MiB"
    elif num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    else:
        return f"{num_bytes} bytes"

def load_all_tensors(model_dir, device="cpu"):
    entries = []
    tensors = {}
    total_bytes = 0

    for fname in os.listdir(model_dir):
        if fname.endswith(".safetensors"):
            full_path = os.path.join(model_dir, fname)
            print(f"{YELLOW}Opening{RESET} {full_path}")

            with safe_open(full_path, framework="pt", device=device) as f:
                for k in f.keys():
                    t = f.get_tensor(k)
                    tensors[k] = t  # store the real tensor

                    num_bytes = t.numel() * t.element_size()
                    total_bytes += num_bytes

                    entries.append((k, list(t.shape), num_bytes))

    # alignment
    max_key_len = max(len(k) for k, _, _ in entries)
    pad = max_key_len + 3

    # print
    for k, shape, num_bytes in entries:
        size_str = format_size(num_bytes)
        print(
            f"{CYAN}{k.ljust(pad)}{RESET} "
            f"{str(shape):20} "
            f"{GREEN}{size_str}{RESET}"
        )

    stats = analyze_weight_sizes(tensors)

    print()
    print(f"{CYAN}Total model size:{RESET}   {GREEN}{format_size(stats['total'])}{RESET}")
    print(f"{CYAN}Embedding layer:{RESET}    {GREEN}{format_size(stats['embed'])}{RESET}")
    print(f"{CYAN}Final norm:{RESET}         {GREEN}{format_size(stats['final_norm'])}{RESET}")
    print(f"{CYAN}Total MLP:{RESET}          {GREEN}{format_size(stats['mlp_total'])}{RESET}")
    print(f"{CYAN}Total Attention:{RESET}    {GREEN}{format_size(stats['attn_total'])}{RESET}")
    print(f"{CYAN}Total LayerNorms:{RESET}   {GREEN}{format_size(stats['layernorm_total'])}{RESET}")
    print()
    layer = stats["layers"][0]

    print(
        f"{YELLOW}Layer {0:2d}:{RESET} "
        f"total={GREEN}{format_size(layer['total'])}{RESET}, "
        f"mlp={GREEN}{format_size(layer['mlp'])}{RESET}, "
        f"attn={GREEN}{format_size(layer['attn'])}{RESET}, "
        f"ln={GREEN}{format_size(layer['layernorm'])}{RESET}"
    )
    return tensors


def main():
    
    model_id = "google/gemma-3-1b-it"
    local_dir = Path.home() / "models" / model_id
    local_dir.mkdir(parents=True, exist_ok=True)


    model_path = snapshot_download(
        model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # actual files, not symlink to cache
    )


    print("Model downloaded to:", model_path)

    load_all_tensors(model_path, device="cpu")

    

if __name__ == "__main__":
    main()
