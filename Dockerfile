FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
# 'libgl1-mesa-glx' is replaced with 'libgl1' for Debian Trixie compatibility.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
# Use tensorflow-cpu to reduce the image size and memory footprint.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make the startup script executable
RUN chmod +x start.sh

# Render uses the PORT environment variable; default to 10000.
EXPOSE 10000

# Run the application
CMD ["./start.sh"]