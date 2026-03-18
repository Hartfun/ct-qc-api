FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data/ data/
COPY train.py .
RUN python train.py  # Creates model/ct_qc_production.pkl

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/model model/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predict.py .

EXPOSE 8000
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]
