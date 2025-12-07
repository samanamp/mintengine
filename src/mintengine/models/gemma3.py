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
    q, k: [L, num_heads, head_dim]
    returns: q_rot, k_rot with the same shape
    """
    L, n_heads, head_dim = q.shape
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    device = q.device
    dtype = q.dtype

    half_dim = head_dim // 2

    # positions: [L]
    positions = torch.arange(L, dtype=dtype, device=device)

    # frequencies: [half_dim]
    # inv_freq[i] = 1 / (rope_theta ** (2i / head_dim))
    idx = torch.arange(half_dim, dtype=dtype, device=device)
    inv_freq = 1.0 / (rope_theta ** (idx / half_dim))

    # freqs: [L, half_dim] = outer(positions, inv_freq)
    freqs = torch.einsum("l,d->ld", positions, inv_freq)

    cos = torch.cos(freqs)  # [L, half_dim]
    sin = torch.sin(freqs)  # [L, half_dim]

    # make them broadcastable over heads: [L, 1, half_dim]
    cos = cos[:, None, :]
    sin = sin[:, None, :]

    def rotate(x):
        # x: [L, num_heads, head_dim]
        x_1 = x[..., :half_dim]  # [L, H, half_dim]
        x_2 = x[..., half_dim:]  # [L, H, half_dim]
        # complex rotation (x_1 + i x_2) * (cos + i sin)
        x_1_rot = x_1 * cos - x_2 * sin
        x_2_rot = x_1 * sin + x_2 * cos
        return torch.cat([x_1_rot, x_2_rot], dim=-1)

    return rotate(q), rotate(k)


class Gemma3Atttention(nn.Module):
    def __init__(self, weights: AttentionWeights):
        super().__init__()
        self.weights: AttentionWeights = weights

    def forward(self, x):
        return x


class Gemma3MLP(nn.Module):
    def __init__(self, weights: MLPWeights):
        super().__init__()
        self.weights: MLPWeights = weights

    def forward(self, x):
        return x


class Gemma3Layer(nn.Module):
    def __init__(self, weights: LayerWeights):
        super().__init__()
        self.weights: LayerWeights = weights
        hidden_size = weights.input_layernorm.shape[0]
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.input_layernorm.weight.data = weights.input_layernorm
        self.attention = Gemma3Atttention(weights.self_attn)
        self.mlp = Gemma3MLP(weights=weights.mlp)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm.weight.data = weights.post_attention_layernorm

        self.pre_feedforward_layernorm = nn.RMSNorm(hidden_size)
        self.pre_feedforward_layernorm.weight.data = weights.pre_feedforward_layernorm

        self.post_feedforward_layernorm = nn.RMSNorm(hidden_size)
        self.post_feedforward_layernorm.weight.data = weights.post_feedforward_layernorm

    def forward(self, x):
        #  Attention
        h1 = self.input_layernorm(x)
        attn_out = self.attention(h1)
        x = x + attn_out

        # post ATTN
        x = self.post_attention_layernorm(x)

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
        self.final_norm = nn.RMSNorm(self.weights.norm.shape[0])
        self.final_norm.weight.data = self.weights.norm
        self.layers = [
            Gemma3Layer(layer_weight) for layer_weight in self.weights.layers
        ]

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embedding_layer(input_ids)
        for layer in self.layers:
            layer(hidden)
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

    def generate(self, text: str, max_tokens: int = 2):
        input_ids = self.tokenizer.encode(text)
        logits = self._generate_from_ids(input_ids, max_tokens)
        return self.tokenizer.decode(logits)

    def _generate_from_ids(self, input_ids: list[int], max_tokens: int):

        result_tensor = torch.IntTensor(input_ids, device=self.device)
        print(result_tensor.unsqueeze(dim=0))
        result_tensor = result_tensor.unsqueeze(dim=0)
        for i in range(max_tokens):
            out = self.decoder(result_tensor)
            print(out)
            out = torch.max(out, dim=1).values
            print(out)
            result_tensor = torch.cat((result_tensor, out), dim=-1)

        return result_tensor
        # print(x.shape)
        # length = x.shape[0]
        # print("length=", length)
        # k_proj_w = self.tensors["model.layers.9.self_attn.k_proj.weight"]
        # [dh, d] = k_proj_w.shape
        # k_layer = nn.Linear(d, dh, bias=False, device=self.device)
        # k_layer.weight.data = k_proj_w
        # k_o = k_layer(x)
        # print(f"{k_o.shape=}")
        # k_norm_layer = nn.RMSNorm(dh)
        # k_norm_layer.weight.data = self.tensors[
        #     "model.layers.9.self_attn.k_norm.weight"
        # ]
        # k_norm_o = k_norm_layer(k_o)
        # print(f"{k_norm_o.shape=}")
        # q_proj_w = self.tensors["model.layers.9.self_attn.q_proj.weight"]
        # [dh, d] = q_proj_w.shape
        # q_layer = nn.Linear(d, dh, bias=False, device=self.device)
        # q_layer.weight.data = q_proj_w
        # q_o = q_layer(x)
        # print(f"{q_o.shape=}")
        # q_o_view = q_o.view(length, 4, int(dh / 4))
        # print(f"{q_o_view.shape=}")
        # q_norm_layer = nn.RMSNorm(int(dh / 4))
        # q_norm_layer.weight.data = self.tensors[
        #     "model.layers.9.self_attn.q_norm.weight"
        # ]
        # q_norm_o = q_norm_layer(q_o_view)
        # print(f"{q_norm_o.shape=}")
        # v_proj_w = self.tensors["model.layers.9.self_attn.v_proj.weight"]
        # [dh, d] = v_proj_w.shape
        # v_layer = nn.Linear(d, dh, bias=False, device=self.device)
        # v_layer.weight.data = v_proj_w
        # v_o = v_layer(x)
        # print(f"{v_o.shape=}")

        # num_kv_heads = 1
        # if num_kv_heads == 1:
        #     k_norm_o = k_norm_o.expand(length, 4, 256)  # [L, 4, 256]
        #     v_o = v_o.expand(length, 4, 256)  # [L, 4, 256]
        # print(f"{v_o.shape=}{k_norm_o.shape=}")
        # rope_theta = 1_000_000
        # Q, K = apply_rope(q_norm_o, k_norm_o, rope_theta=rope_theta)

        # # 7. Reorder to [batch, heads, seq, dim] for attention math
        # # here batch size is 1

        # q_norm_o = q_norm_o.permute(1, 0, 2).unsqueeze(0)  # [1, 4, L, 256]
        # k_norm_o = k_norm_o.permute(1, 0, 2).unsqueeze(0)  # [1, 4, L, 256]
        # v_o = v_o.permute(1, 0, 2).unsqueeze(0)  # [1, 4, L, 256]

        # qk = torch.matmul(q_norm_o, k_norm_o.transpose(-1, -2))
        # scores = qk / sqrt(256)
        # print(f"{scores.shape=}")
        # weights = torch.softmax(scores, dim=-1)
        # print(f"{weights.shape=}")
        # context = torch.matmul(weights, v_o)
        # print(f"{context.shape=}")

        # context = context.squeeze(0).permute(1, 0, 2)
        # print("context", context.shape)
        # context = context.reshape(length, 4 * 256)
        # print("context", context.shape)
        # o_proj_w = tensors["model.layers.9.self_attn.o_proj.weight"]
        # [dh, d] = o_proj_w.shape
        # o_layer = nn.Linear(d, dh, bias=False, device=device)
        # o_layer.weight.data = o_proj_w
        # att_proj_o = o_layer(context)
        # print(f"{att_proj_o.shape=}")
        # hidden = att_proj_o
