# syntax=docker/dockerfile:1
# Base image pinned to python:3.12-slim SHA (2026-07-30 digest; update periodically).
# To re-pin: docker pull python:3.12-slim && docker inspect python:3.12-slim --format='{{index .RepoDigests 0}}'
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS builder

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY agent/       ./agent/
COPY classifier/  ./classifier/
COPY data/        ./data/
COPY logger/      ./logger/
COPY planner/     ./planner/
COPY static/      ./static/
COPY tools/       ./tools/
COPY app.py       .

# Create logs directory (runtime trace output) and drop root privileges
RUN mkdir -p logs && \
    addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Default port (override with -e PORT=xxxx at runtime)
ENV PORT=5001
ENV PYTHONPATH=/app

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["sh", "-c", "python3 app.py"]
