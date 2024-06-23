FROM python:3.12-slim

COPY requirements.txt requirements.lock /
COPY app /app

RUN ls /
RUN ls /app

RUN pip install --no-cache-dir -r /requirements.txt -c /requirements.lock

EXPOSE 8080
ENTRYPOINT [ "python", "/app/main.py" ]
