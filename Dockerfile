# Base image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Chrome
RUN apt-get update && apt-get install -y \
    wget gnupg unzip curl xvfb \
    chromium chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY .env .

# Create downloads directory
RUN mkdir -p /app/downloads

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

EXPOSE 7860

# Start Xvfb and run app
CMD Xvfb :99 -screen 0 1920x1080x24 & python main.py
