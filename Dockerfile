FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    lancedb \
    pyarrow \
    python-multipart

# Copy code
COPY chat_frontend.py /app/
COPY chat_app.py /app/

# Create data directory
RUN mkdir -p /data/chat_lancedb

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "chat_app:app", "--host", "0.0.0.0", "--port", "8080"]
