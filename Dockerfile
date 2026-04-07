# Dockerfile — SRE Incident Response OpenEnv
# Compatible with Hugging Face Spaces (Docker SDK)
# Build: docker build -t sre-incident-env .
# Run:   docker run -p 7860:7860 -e HF_TOKEN=... sre-incident-env

FROM python:3.11-slim

# HF Spaces requires a non-root user with uid=1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"
ENV HOME="/home/user"
ENV PYTHONPATH="/home/user/app"
ENV PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY --chown=user . .

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check — curl is not in slim; use Python urllib instead
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read()"

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
