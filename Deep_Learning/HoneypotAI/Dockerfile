# HoneypotAI - Advanced Adaptive Cybersecurity Intelligence Platform
# Multi-stage build for optimized production image

FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HONEYPOT_LOG_LEVEL=INFO \
    HONEYPOT_CONFIG_PATH=/app/config/honeypot.yaml

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Create non-root user
RUN groupadd -r honeypotai && \
    useradd -r -g honeypotai honeypotai

# Copy application code
COPY --chown=honeypotai:honeypotai . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/config /app/data && \
    chown -R honeypotai:honeypotai /app

# Switch to non-root user
USER honeypotai

# Expose ports
EXPOSE 8501 2222 8080 2121

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["python", "main.py", "--start-honeypot"]

# Labels
LABEL maintainer="HoneypotAI Team <support@honeypotai.com>" \
      version="1.0.0" \
      description="Advanced Adaptive Cybersecurity Intelligence Platform" \
      org.opencontainers.image.title="HoneypotAI" \
      org.opencontainers.image.description="Next-generation cybersecurity intelligence platform with AI-powered threat detection" \
      org.opencontainers.image.vendor="HoneypotAI" \
      org.opencontainers.image.licenses="MIT"
