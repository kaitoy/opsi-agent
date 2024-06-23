FROM python:3.12.3-slim

COPY requirements.txt requirements.lock /

RUN apt-get update \
    && \
    apt-get install -y --no-install-recommends \
      libpq5 \
    && \
    apt-get -y clean \
    && \
    rm -rf /var/lib/apt/lists/* \
    && \
    pip install --no-cache-dir -r /requirements.txt -c /requirements.lock

COPY app /app

EXPOSE 8080
ENTRYPOINT [ "python", "/app/main.py" ]
