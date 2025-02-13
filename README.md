1. Install library
```
pip install uv
uv venv
. .venv/bin/activate
uv pip install git+https://github.com/condenses/subnet-scoring.git
uv pip install flash-attn --no-build-isolation
export HF_HUB_ENABLE_HF_TRANSFER=1
```

2. Run server (FastAPI)
- Env available:
    - `CONDENSES_SCORING_HOST`
    - `CONDENSES_SCORING_PORT`
```
uv run -m condenses_scoring.server
```

3. Run tests
```
pytest
```




