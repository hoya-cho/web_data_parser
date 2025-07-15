# Use PyTorch CUDA base image
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

# Install git and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    # For OpenCV and other graphics/display libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # For Selenium/web scraping (headless browser dependencies)
    libnss3 \
    libatk-bridge2.0-0 \
    libxss1 \
    libgbm1 \
    libgtk-3-0 \
    libasound2 \
    # Additional Chrome and ChromeDriver dependencies
    wget \
    unzip \
    libappindicator1 \
    libindicator7 \
    fonts-liberation \
    libdrm-dev \
    # Clean up apt lists to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/parser_service_updated_prompts/

# Set the working directory to the parser_service_updated_prompts directory
WORKDIR /app/parser_service_updated_prompts

# Set PYTHONPATH to include the parent directory
ENV PYTHONPATH=/app:$PYTHONPATH

# Set Hugging Face token and login

# Make port 8501 available (Streamlit default port)
EXPOSE 8501

# Command to run the Streamlit application
CMD ["python", "-m", "streamlit", "run", "web_demo.py"] 