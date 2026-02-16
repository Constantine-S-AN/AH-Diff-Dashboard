FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY data ./data
COPY docs ./docs
COPY scripts ./scripts

RUN pip install --upgrade pip && \
    pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/ah_premium_lab/app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
