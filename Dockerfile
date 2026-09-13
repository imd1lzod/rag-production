FROM python:3.12-slim

RUN groupadd -r appuser && useradd -r -g appuser --create-home appuser

WORKDIR /app

RUN chown -R appuser:appuser /app

USER appuser

RUN pip install --user uv

ENV PATH="/home/appuser/.local/bin:$PATH"

COPY --chown=appuser:appuser pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY --chown=appuser:appuser app/ app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]