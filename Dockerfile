# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Run main.py to generate model.pkl
RUN python main.py

# Expose ports
EXPOSE 5000 8000

# Use JSON format for CMD to prevent unintended behavior related to OS signals
CMD ["sh", "-c", "mlflow ui --host 0.0.0.0 --port 5000 & uvicorn app:app --host 0.0.0.0 --port 8000"]

