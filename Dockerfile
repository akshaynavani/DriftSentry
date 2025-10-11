FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# CPU-only torch wheel keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY driftsentry ./driftsentry
RUN pip install --no-cache-dir .

# Mount or bake a trained bundle here (output of `driftsentry train --out artifacts`)
ENV DRIFTSENTRY_ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000

CMD ["uvicorn", "driftsentry.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
