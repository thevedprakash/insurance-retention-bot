# Use the official lightweight Python image.
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Copy the requirements file into the container
COPY --chown=appuser:appuser requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY --chown=appuser:appuser . /app

# Expose the port the app runs on
EXPOSE 5000

# Set the entry point to run the Flask application
ENTRYPOINT ["python", "app.py"]
