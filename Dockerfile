FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000 8000

CMD mlflow ui --host 0.0.0.0 --port 5000 & uvicorn app:app --host 0.0.0.0 --port 8000
