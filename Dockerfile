# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY realtime_trading_bot.py .

# Create non-root user for security
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# Run the bot
CMD ["python", "-u", "realtime_trading_bot.py"]
