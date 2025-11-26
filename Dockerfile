FROM python:3.11-slim

# Workdir inside the container
WORKDIR /app

# System deps for mujoco rendering
RUN apt-get update && apt-get install -y \
    libgl1 libglfw3 libglew2.2 patchelf && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command: run the pick-and-place script
CMD ["python", "pnp.py"]
