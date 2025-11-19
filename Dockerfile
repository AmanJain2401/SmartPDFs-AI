# ---------- Base Image ----------
# Use Python 3.9 slim image
FROM python:3.9-slim

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Copy Requirements ----------
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Project Files ----------
COPY . .

# ---------- Environment ----------
# Streamlit headless mode for Docker
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PYTHONUNBUFFERED=1

# Optional: expose port
EXPOSE 8501

# ---------- Run App ----------
CMD ["streamlit", "run", "app.py"]
