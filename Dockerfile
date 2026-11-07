# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv and the app code.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 demandcast
WORKDIR /app
COPY --from=build --chown=demandcast:demandcast /app /app
RUN mkdir -p /app/data && chown demandcast:demandcast /app/data
USER demandcast
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8824 DEMANDCAST_DATA_DIR=/app/data
EXPOSE 8824
HEALTHCHECK --interval=15s --timeout=3s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8824/healthz', timeout=2)"
# Train on first start if the data volume is empty, then serve.
CMD ["sh", "-c", "[ -f /app/data/artifacts.json ] || demandcast --data-dir /app/data train; exec demandcast --data-dir /app/data serve"]
