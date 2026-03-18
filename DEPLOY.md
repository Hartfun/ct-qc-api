# CT QC Anomaly Detection — Render Docker Deployment

## Repo layout expected by the Dockerfile

```
.
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── Dockerfile
├── data/
│   └── CT-Test.csv          # training data (local only, not committed)
└── model/
    └── ct_qc_production.pkl # MUST be committed — built by train.py
```

---

## 1 — Train the model locally (one-time)

```bash
pip install -r requirements.txt
python train.py
# → writes model/ct_qc_production.pkl
git add model/ct_qc_production.pkl
git commit -m "add trained model"
git push
```

## 2 — Build & test the Docker image locally

```bash
docker build -t ct-qc .
docker run --rm -p 10000:10000 ct-qc
# Open http://localhost:10000/docs
```

## 3 — Deploy on Render

1. Go to https://dashboard.render.com → **New → Web Service**
2. Connect your GitHub repo
3. Set **Environment** to `Docker`
4. Render auto-detects the `Dockerfile` — no extra config needed
5. Under **Environment Variables** you can override `PORT` (Render injects it automatically)
6. Click **Deploy**

Render will:
- Run `docker build` using your `Dockerfile`
- Start the container; the `HEALTHCHECK` on `/health` confirms readiness

---

## API usage

### POST /predict

```bash
curl -X POST https://<your-app>.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "serial_No": "CT-001",
    "Slice thickness 1.5": 1.5,
    "Slice thickness 5": 5.0,
    "Slice thickness 10": 10.0,
    "KV accuracy 80": 80.0,
    "KV accuracy 110": 110.0,
    "KV accuracy 130": 130.0,
    "Accuracy Timer 0.8": 0.8,
    "Accuracy Timer 1": 1.0,
    "Accuracy Timer 1.5": 1.5,
    "Radiation Dose Test (Head) 21.50": 21.50,
    "Radiation Dose Test (Body) 10.60": 10.60,
    "Low Contrast Resolution 5.0": 5.0,
    "High Contrast Resolution 6.24": 6.24,
    "Radiation Leakage Levels (Front)": 50.0,
    "Radiation Leakage Levels (Back)": 40.0,
    "Radiation Leakage Levels (Left)": 45.0,
    "Radiation Leakage Levels (Right)": 55.0
  }'
```

---

## Fixes applied (summary)

| File | Issue | Fix |
|------|-------|-----|
| `app.py` | `pass` after `return` in `preprocess` | Removed |
| `app.py` | Pydantic v2 deprecated `class Config` | Replaced with `model_config = {...}` |
| `app.py` | `.dict()` deprecated in Pydantic v2 | Replaced with `.model_dump(by_alias=True)` |
| `app.py` | `get_model()` opened file without explicit close | Used `with open(...)` |
| `app.py` | `HTTPException` swallowed in `/predict` | Re-raised explicitly |
| `app.py` | `PORT` hardcoded to 10000 in `__main__` | Read from `os.environ.get("PORT", 10000)` |
| `app.py` | CORS middleware declared but never added | Moved `add_middleware` call after `app` init |
| `Dockerfile` | `curl` missing in slim image → `HEALTHCHECK` always fails | `apt-get install curl` added |
| `Dockerfile` | `CMD` used exec form `["uvicorn", ...]` — `$PORT` not expanded | Changed to shell form |
| `Dockerfile` | No `--start-period` on `HEALTHCHECK` | Added `--start-period=15s` |
| `predict.py` | `from train import ...` — circular/missing import at deploy time | Inlined all constants, removed train import |
| `train.py` | `reset_index(drop=True)` missing after `sort_values` | Added |
