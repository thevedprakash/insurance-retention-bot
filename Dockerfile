FROM python:3.10-slim

ENV PYTHONUNBUFFERED=True

WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

COPY ./requirements.txt ./requirements.txt

# Use --no-cache-dir to avoid caching packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Set the entry point to run the Flask application
ENTRYPOINT ["python", "run.py"]

EXPOSE 8000