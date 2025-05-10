# Use an official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Copy only required files first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose port
EXPOSE 8000

# Default command to run the API
CMD ["uvicorn", "app.routes.main:app", "--host", "0.0.0.0", "--port", "8000"]
