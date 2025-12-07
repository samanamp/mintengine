# mintengine

Install mintengine:
```
uv sync
uv pip install -e .
```

Run mintengine:
```
uv run mintengine

For accessing gated model
```
HF_TOKEN=hf_xxxxx uv run mintengine
```

For windows
```
$env:HF_TOKEN = "hf_xxxxx"
uv run mintengine
```

## References
We are using the following references for architecture and code correctness.

[vLLM - Add support for Gemma 3](https://github.com/vllm-project/vllm/pull/14660)