# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY . .

# Set environment variables for configuration
ENV CONFIG_FILE=/app/config.py \
    PROMETHEUS_GATEWAY=http://pushgateway:9091

# Expose the port on which the application will run (if applicable)
# EXPOSE 8000

# Run the forecast generation script when the container starts
CMD ["python", "dva_forecast_generation.py"]

