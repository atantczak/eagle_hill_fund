# Eagle Hill Fund site: FastAPI + Jinja over S3, plus the lab over an S3 mirror of
# the Studio's var/ (ADR-0009). Built by .github/workflows/site.yml from the
# repository root:  docker build -f infra/aws/site/Dockerfile .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    EHF_VAR_DIR=/var/ehf

WORKDIR /app

COPY infra/aws/site/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# The whole package (tests, notebooks and caches are kept out by .dockerignore):
# ehf.web serves the site, ehf.research/integrations/tools run the lab.
COPY ehf ehf

# The only writable place at runtime (an ephemeral task volume is mounted here and
# inherits this directory's owner); the root filesystem is read-only.
RUN useradd --system --uid 10001 --no-create-home app \
    && mkdir -p /var/ehf && chown app:app /var/ehf
USER app

EXPOSE 8080
CMD ["uvicorn", "ehf.web.app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips=*", "--no-server-header"]
