# Use a slim Python 3.11 base image for efficiency and safety
FROM python:3.11-slim

# Prevent Python from writing pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system dependencies if any are needed in the future (none for now)
# Copy and install python dependencies first to cache this layer
COPY requirements.txt /workspace/vector_engine/requirements.txt
RUN pip install --no-cache-dir -r /workspace/vector_engine/requirements.txt

# Copy the rest of the application
COPY . /workspace/vector_engine

# Expose FastAPI's default port
EXPOSE 8000

# Set Python path so that uvicorn can find vector_engine
ENV PYTHONPATH=/workspace

# Run the FastAPI service on container startup
CMD ["uvicorn", "vector_engine.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
