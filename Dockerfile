FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio:9000

CMD ["python", "src/app.py"]
