# Use an official Ubuntu image as a base
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
# Add /root/.local/bin to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies, including Git
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Default command
CMD ["bash"]
