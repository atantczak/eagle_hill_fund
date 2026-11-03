# Eagle Hill Fund site: FastAPI + Jinja over S3. Built by .github/workflows/site.yml
# from the repository root:  docker build -f infra/aws/site/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY infra/aws/site/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Only the web package and the package marker above it; nothing from research/.
COPY ehf/__init__.py ehf/__init__.py
COPY ehf/web ehf/web

RUN useradd --system --uid 10001 --no-create-home app
USER app

EXPOSE 8080
CMD ["uvicorn", "ehf.web.app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips=*", "--no-server-header"]
