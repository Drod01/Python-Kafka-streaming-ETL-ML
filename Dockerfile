FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY scripts ./scripts

RUN useradd --create-home --uid 10001 wellstream \
    && mkdir -p /app/models /app/artifacts /app/data \
    && chown -R wellstream:wellstream /app
USER wellstream

ENTRYPOINT ["wellstream"]
CMD ["--help"]
