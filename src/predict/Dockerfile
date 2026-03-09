FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN pip install --no-cache-dir uv && \
    uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.predict.services.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
