# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Create a virtual environment and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -e . && \
    uv pip install --upgrade pip    

# Copy the rest of the application code
COPY src ./src

ENTRYPOINT ["/bin/bash", "-c", "source .venv/bin/activate && exec $0 $@"]

# Run the application. Train the model.
CMD ["python", "src/train.py"]