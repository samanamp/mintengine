# inferlite-engine

Install inferlite:
```
uv sync
uv pip install -e .
```

Run Inferlite:
```
uv run inferlite

For accessing gated model
```
HF_TOKEN=hf_xxxxx uv run inferlite
```

For windows
```
$env:HF_TOKEN = "hf_xxxxx"
uv run inferlite
```

## References
We are using the following references for architecture and code correctness.

[vLLM - Add support for Gemma 3](https://github.com/vllm-project/vllm/pull/14660)