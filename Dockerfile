FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
COPY app/viewer_requirements.txt ./app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r app/viewer_requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# Copy application code
COPY . .
RUN pip install -e .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start script will be provided by docker-compose
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
