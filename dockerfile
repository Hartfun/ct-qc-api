FROM python:3.12-slim

WORKDIR /app

# FIX: install curl for the HEALTHCHECK before copying app files
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# FIX: guard against missing model dir — train.py must be run before building
# The COPY will fail at build time if the file is absent, which is the correct behaviour.
COPY model/ct_qc_production.pkl model/ct_qc_production.pkl

# Render injects PORT at runtime; default to 10000 to match their free-tier convention
ENV PORT=10000
EXPOSE ${PORT}

# FIX: use shell form so $PORT is expanded correctly at runtime
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# FIX: use shell form (not exec form) so $PORT env-var is expanded
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
