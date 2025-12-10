import torch
from vllm import LLM, SamplingParams

MODEL = "google/gemma-3-1b-it"

# vLLM params to expose internals
llm = LLM(
    model=MODEL,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    # These two flags expose layer-wise outputs
    return_logits=True,
    return_all_hidden_states=True,  # <--- important
)

sampling = SamplingParams(
    temperature=0.0,
)

prompt = "Hello world."

outputs = llm.generate(
    [prompt],
    sampling_params=sampling,
)

out = outputs[0]

# ---------------------------------------------------------
# Hidden state structure:
# out.outputs[i].hidden_states[layer_id][token]
# Shape: [num_layers+1][seq_len][hidden_dim]
# ---------------------------------------------------------

hidden_states = out.outputs[0].hidden_states

print(f"Total layers (including embeddings): {len(hidden_states)}")
print(f"Sequence length: {hidden_states[0].shape[0]}")
print(f"Hidden size: {hidden_states[0].shape[1]}")

# Print layer-by-layer statistics
for layer_idx, h in enumerate(hidden_states):
    h_tensor = torch.tensor(h)
    print(
        f"Layer {layer_idx:02d}: "
        f"shape={h_tensor.shape}, "
        f"mean={h_tensor.mean().item():.6f}, "
        f"std={h_tensor.std().item():.6f}"
    )

# Example: access layer N, token T
L = 5  # 5th layer block output
T = -1  # last token
vec = torch.tensor(hidden_states[L][T])
print(f"\nLayer {L} last token vector:\n{vec[:10]}")
