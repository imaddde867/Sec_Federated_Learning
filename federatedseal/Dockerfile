FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir tenseal

# Copy your scripts
COPY federated-tenseal.py .
COPY tenseal_adapter.py .

# Default command
CMD ["python", "federated-tenseal.py"]
