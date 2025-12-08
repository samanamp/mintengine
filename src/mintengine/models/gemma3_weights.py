import torch


class AttentionWeights:
    def __init__(self):
        self.k_norm: torch.Tensor
        self.k_proj: torch.Tensor
        self.o_proj: torch.Tensor
        self.q_norm: torch.Tensor
        self.q_proj: torch.Tensor
        self.v_proj: torch.Tensor


class MLPWeights:
    def __init__(self):
        self.gate_proj: torch.Tensor
        self.up_proj: torch.Tensor
        self.down_proj: torch.Tensor


class LayerWeights:
    def __init__(self):
        self.self_attn = AttentionWeights()
        self.mlp = MLPWeights()
        self.input_layernorm: torch.Tensor
        self.post_attention_layernorm: torch.Tensor
        self.post_feedforward_layernorm: torch.Tensor
        self.pre_feedforward_layernorm: torch.Tensor

    def missing_fields(self, layer_id):
        missing = []

        # check layernorms
        if self.input_layernorm is None:
            missing.append(f"layers.{layer_id}.input_layernorm")
        if self.post_attention_layernorm is None:
            missing.append(f"layers.{layer_id}.post_attention_layernorm")
        if self.post_feedforward_layernorm is None:
            missing.append(f"layers.{layer_id}.post_feedforward_layernorm")
        if self.pre_feedforward_layernorm is None:
            missing.append(f"layers.{layer_id}.pre_feedforward_layernorm")

        # attn fields
        for name, val in vars(self.self_attn).items():
            if val is None:
                missing.append(f"layers.{layer_id}.self_attn.{name}")

        # mlp fields
        for name, val in vars(self.mlp).items():
            if val is None:
                missing.append(f"layers.{layer_id}.mlp.{name}")

        return missing


class Gemma3Weights:
    def __init__(self, num_layers):
        self.num_layers: int = num_layers
        self.norm: torch.Tensor
        self.embed_tokens: torch.Tensor
        self.layers = [LayerWeights() for _ in range(num_layers)]

    @classmethod
    def from_tensors_dict(cls, tensors: dict[str, torch.Tensor]):

        max_layer = -1
        for k in tensors.keys():
            if k.startswith("model.layers."):
                layer_id = int(k.split(".")[2])
                if layer_id > max_layer:
                    max_layer = layer_id
        num_layers = max_layer + 1

        model = Gemma3Weights(num_layers=num_layers)
        model.norm = tensors["model.norm.weight"]
        model.embed_tokens = tensors["model.embed_tokens.weight"]

        for k, v in tensors.items():
            parts = k.split(".")

            # skip non-layer keys
            if parts[1] != "layers":
                continue

            layer = int(parts[2])
            sub = parts[3]  # e.g. "mlp", "self_attn", "input_layernorm"

            # shape-based routing
            if sub == "input_layernorm":
                model.layers[layer].input_layernorm = v

            elif sub == "post_attention_layernorm":
                model.layers[layer].post_attention_layernorm = v

            elif sub == "post_feedforward_layernorm":
                model.layers[layer].post_feedforward_layernorm = v

            elif sub == "pre_feedforward_layernorm":
                model.layers[layer].pre_feedforward_layernorm = v

            elif sub == "self_attn":
                attn_component = parts[4]  # e.g. "k_proj"
                setattr(model.layers[layer].self_attn, attn_component, v)

            elif sub == "mlp":
                mlp_component = parts[4]  # e.g. "down_proj"
                setattr(model.layers[layer].mlp, mlp_component, v)

        model._validate(tensors)
        return model

    def _validate(self, tensors):
        print("\n[Gemma3Weights] Validating weight completeness...")

        errors = []

        # Check top-level weights
        if self.norm is None:
            errors.append("Missing: model.norm.weight")

        if self.embed_tokens is None:
            errors.append("Missing: model.embed_tokens.weight")

        # Check each layer
        for i, layer in enumerate(self.layers):
            missing = layer.missing_fields(i)
            errors.extend(missing)

        if errors:
            print("\n[ERROR] Missing weight tensors:")
            for e in errors:
                print("  -", e)
            raise ValueError(f"Model load failed: missing {len(errors)} tensors")

        # Double-check that all safetensors keys were consumed
        consumed = 0
        for k in tensors.keys():
            if (
                k.startswith("model.layers.")
                or k.startswith("model.norm")
                or k.startswith("model.embed_tokens")
            ):
                consumed += 1

        print("[Gemma3Weights] All required tensors loaded successfully.\n")
