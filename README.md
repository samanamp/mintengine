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