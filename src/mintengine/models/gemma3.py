from math import sqrt
import torch
import torch.nn as nn
from mintengine.models.gemma3_weights import (
    AttentionWeights,
    Gemma3Weights,
    LayerWeights,
    MLPWeights,
)
from mintengine.models.helpers import detect_device, get_hf_model
from mintengine.models.weight_loader import load_all_tensors
import sentencepiece as spm


def apply_rope(q, k, rope_theta=1000000.0):
    """
    q, k: [batch, num_heads, L, head_dim]
    returns: q_rot, k_rot with the same shape
    """
    batch, n_heads, L, head_dim = q.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    device = q.device
    dtype = q.dtype

    half_dim = head_dim // 2

    # positions: [L]
    positions = torch.arange(L, dtype=dtype, device=device)

    # frequencies: [half_dim]
    # inv_freq[i] = 1 / (rope_theta ** (2i / head_dim))
    idx = torch.arange(half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (rope_theta ** (2 * idx / head_dim))

    # freqs: [L, half_dim] = outer(positions, inv_freq)
    freqs = torch.einsum("l,d->ld", positions, inv_freq)

    cos = torch.cos(freqs)  # [L, half_dim]
    sin = torch.sin(freqs)  # [L, half_dim]

    # make them broadcastable over heads: [L, 1, half_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    def rotate(x):
        # x: [B, L, num_heads, head_dim]
        x_1 = x[..., :half_dim]  # [B, L, H, half_dim]
        x_2 = x[..., half_dim:]  # [B, L, H, half_dim]
        # complex rotation (x_1 + i x_2) * (cos + i sin)
        x_1_rot = x_1 * cos - x_2 * sin
        x_2_rot = x_1 * sin + x_2 * cos
        return torch.cat([x_1_rot, x_2_rot], dim=-1)

    return rotate(q), rotate(k)


class Gemma3Atttention(nn.Module):
    def __init__(self, weights: AttentionWeights, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.weights: AttentionWeights = weights

        self.num_kv_heads = 1  # Gemma uses 1 KV head
        k_proj_w = weights.k_proj
        [dh, d] = k_proj_w.shape
        self.device = k_proj_w.device
        self.head_dim = dh
        self.k_layer = nn.Linear(d, dh, bias=False, device=self.device)
        self.k_layer.weight.data = k_proj_w

        self.k_norm_layer = nn.RMSNorm(dh, eps=1e-6)
        self.k_norm_layer.weight.data = weights.k_norm

        q_proj_w = weights.q_proj
        [dh, d] = q_proj_w.shape
        self.num_heads = int(dh / self.head_dim)
        # print(f"{self.num_heads=}")
        self.q_layer = nn.Linear(d, dh, bias=False, device=self.device)
        self.q_layer.weight.data = q_proj_w

        self.q_norm_layer = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.q_norm_layer.weight.data = weights.q_norm

        v_proj_w = weights.v_proj
        [dh, d] = v_proj_w.shape
        self.v_layer = nn.Linear(d, dh, bias=False, device=self.device)
        self.v_layer.weight.data = v_proj_w

        o_proj_w = weights.o_proj
        [dh, d] = o_proj_w.shape
        self.o_layer = nn.Linear(d, dh, bias=False, device=self.device)
        self.o_layer.weight.data = o_proj_w

    def forward(self, x):
        # print(f"{x.shape=}")
        length = x.shape[-2]
        batch = x.shape[0]
        # print("length=", length)

        k_o = self.k_layer(x)
        # print(f"{k_o.shape=}")
        k_o = self.k_norm_layer(k_o)
        # print(f"{k_o.shape=}")

        q_o = self.q_layer(x)
        # print(f"{q_o.shape=}")
        q_o_view = q_o.view(batch, length, self.num_heads, self.head_dim)
        # print(f"{q_o_view.shape=}")

        q_o = self.q_norm_layer(q_o_view)
        q_o = q_o.permute(0, 2, 1, 3)  # [batch, num_heads, L, head_dim]
        # print(f"*{q_o.shape=}")

        v_o = self.v_layer(x)
        # print(f"{v_o.shape=}")

        # if self.layer_id == 25:
        #     print(f"v_o IMMEDIATELY after v_layer:")
        #     print(f"  shape: {v_o.shape}")
        #     print(f"  pos 0: {v_o[0, 0, :5]}")
        #     print(f"  pos 1: {v_o[0, 1, :5]}")
        #     print(f"  pos 2: {v_o[0, -1, :5]}")

        num_kv_heads = 1
        k_o = k_o.view(batch, length, num_kv_heads, self.head_dim).transpose(1, 2)
        v_o = v_o.view(batch, length, num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads < self.num_heads:
            k_o = k_o.expand(
                batch, self.num_heads, length, self.head_dim
            )  # [batch, num_heads, L, head_dim]
            v_o = v_o.expand(
                batch, self.num_heads, length, self.head_dim
            )  # [batch, num_heads, L, head_dim]
        # print(f"*{v_o.shape=}\n*{k_o.shape=}")
        rope_theta = 1_000_000
        q_o, k_o = apply_rope(q_o, k_o, rope_theta=rope_theta)

        # 7. Reorder to [batch, heads, seq, dim] for attention math
        # here batch size is 1

        # q_o = q_o.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads, L, head_dim]
        # k_o = k_o.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads, L, head_dim]
        # v_o = v_o.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads, L, head_dim]

        qk = torch.matmul(q_o, k_o.transpose(-1, -2))
        scores = qk / sqrt(256)

        # Add causal mask
        mask = torch.triu(
            torch.ones(length, length, device=self.device, dtype=scores.dtype),
            diagonal=1,
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))

        scores = scores + mask  # Broadcast over batch and heads

        # print(f"{scores.shape=}")
        # print(f"Mask:\n{mask[:5, :5]}")
        weights = torch.softmax(scores, dim=-1)
        # print(f"{weights.shape=}")

        context = torch.matmul(weights, v_o)
        # print(f"{context.1shape=}")
        # if self.layer_id == 25:
        # print(f"k_o shape: {k_o.shape}")
        # print(f"k_o[0, 0, 0, :5] (pos 0): {k_o[0, 0, 0, :5]}")
        # print(f"k_o[0, 0, 1, :5] (pos 1): {k_o[0, 0, 1, :5]}")
        # print(f"k_o[0, 0, 2, :5] (pos 2): {k_o[0, 0, 2, :5]}")
        # print(f"k_o[0, 0, -1, :5] (last pos): {k_o[0, 0, -1, :5]}")
        # print(f"Attention weights for last token: {weights[0, 0, -1, :]}")
        # print(f"context: {context[0, 0, -1, :]}")

        context = context.permute(0, 2, 1, 3)
        # print("context", context.shape)
        context = context.reshape(batch, length, self.num_heads * self.head_dim)
        # print("context", context.shape)

        att_proj_o = self.o_layer(context)
        # print(f"{att_proj_o.shape=}")
        x = att_proj_o
        return x


class Gemma3MLP(nn.Module):
    def __init__(self, weights: MLPWeights):
        super().__init__()
        self.weights: MLPWeights = weights
        [self.hidden_size, self.intermediate_size] = weights.down_proj.shape
        self.gate_up_proj_w = torch.concat(
            (self.weights.gate_proj, self.weights.up_proj)
        ).T
        self.activation_func = nn.GELU(approximate="tanh")

    def forward(self, x):
        # MLP(x) = W_down( GELU(W_gate x) * (W_up x) )
        gate_up_proj = torch.matmul(x, self.gate_up_proj_w)
        g, u = torch.split(gate_up_proj, self.intermediate_size, dim=-1)
        g = self.activation_func(g)
        x = torch.matmul(g * u, self.weights.down_proj.T)
        return x


class Gemma3Layer(nn.Module):
    def __init__(self, weights: LayerWeights, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.weights: LayerWeights = weights

        hidden_size = weights.input_layernorm.shape[0]
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.input_layernorm.weight.data = weights.input_layernorm
        self.attention = Gemma3Atttention(weights.self_attn, layer_id)
        self.mlp = Gemma3MLP(weights=weights.mlp)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm.weight.data = weights.post_attention_layernorm

        self.pre_feedforward_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.pre_feedforward_layernorm.weight.data = weights.pre_feedforward_layernorm

        self.post_feedforward_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_feedforward_layernorm.weight.data = weights.post_feedforward_layernorm

    def forward(self, x):

        #  Attention
        h1 = self.input_layernorm(x)
        if self.layer_id == 1:
            print("x-layer1", x.shape, x[0, :, 0:5])
        if self.layer_id == 2:
            print("x-layer2", x.shape, x[0, :, 0:5])
        attn_out = self.attention(h1)
        x = x + attn_out
        # if self.layer_id == 25:
        #     print("post attention", attn_out)
        # post ATTN
        x = self.post_attention_layernorm(x)

        # if self.layer_id == 8:
        #     print("x-layer8", x.shape, x[0, :, 0:5])
        # Feed Forward (MLP)
        h2 = self.pre_feedforward_layernorm(x)
        ff_out = self.mlp(h2)
        x = x + ff_out

        # Post FFN
        hidden = self.post_feedforward_layernorm(x)

        return hidden


class Gemma3Decoder(nn.Module):
    def __init__(self, weights: Gemma3Weights):
        super().__init__()
        self.weights = weights

        self.embedding_layer = nn.Embedding.from_pretrained(self.weights.embed_tokens)
        self.final_norm = nn.RMSNorm(self.weights.norm.shape[0], eps=1e-6)
        self.final_norm.weight.data = self.weights.norm
        self.layers = nn.ModuleList(
            [
                Gemma3Layer(layer_weight, i)
                for i, layer_weight in enumerate(self.weights.layers)
            ]
        )

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embedding_layer(input_ids)

        hidden = hidden * (self.weights.embed_tokens.shape[1] ** 0.5)
        # print(f"{hidden=}")
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        logits = torch.matmul(hidden, self.weights.embed_tokens.T)
        return logits


class Gemma3:
    def __init__(self):
        self.device = detect_device()
        self.model_path = get_hf_model(model_id="google/gemma-3-1b-it")
        self.tensors = load_all_tensors(self.model_path, device=self.device)
        self.gemma3_weights = Gemma3Weights.from_tensors_dict(self.tensors)
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=self.model_path + "/tokenizer.model"
        )
        self.decoder = Gemma3Decoder(self.gemma3_weights)

    def generate(self, text: str, max_tokens: int = 1):
        input_ids = self.tokenizer.encode(text)
        result_ids = self._generate_from_ids(input_ids, max_tokens)
        print(f"{result_ids=}")
        return self.tokenizer.decode(result_ids[0].tolist())

    @torch.no_grad()
    def _generate_from_ids(self, input_ids: list[int], max_tokens: int):

        result_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        print(result_tensor.unsqueeze(dim=0))
        result_tensor = result_tensor.unsqueeze(dim=0)
        for i in range(max_tokens):
            out = self.decoder(result_tensor)
            # print(out)
            logits = out[0, -1, :]
            # print(f"{logits=}")
            # next_token = torch.argmax(out[0, -1, :])
            # Add temperature sampling
            temperature = 0.7
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            print(f"{next_token=}")
            result_tensor = torch.cat((result_tensor, next_token.unsqueeze(0)), dim=1)
            # print(f"{result_tensor=}")

        return result_tensor
