FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model/ct_qc_production.pkl .

ENV PORT=10000
EXPOSE $PORT

HEALTHCHECK --interval=30s CMD curl -f http://localhost:$PORT/health || exit 1

CMD uvicorn app:app --host 0.0.0.0 --port $PORT
