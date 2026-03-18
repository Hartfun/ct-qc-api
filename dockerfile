FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN python -m venv /opt/venv \
  && . /opt/venv/bin/activate \
  && pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY train.py .
RUN . /opt/venv/bin/activate && python train.py  # Creates model/ct_qc_production.pkl

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/model model/
COPY predict.py .

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE $PORT
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
