# Use Python 3.9 as the base image
FROM python:3.9

# Expose port 8080
EXPOSE 8080

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Update the package manager and install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0

# Install TensorFlow and other Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . /app

# Optional: Run any additional setup scripts
RUN /bin/sh setup.sh

# Define the entry point for the container
ENTRYPOINT ["streamlit", "run", "Demo.py"]

