FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY src/model.py ./src/model.py
COPY src/__init__.py ./src/__init__.py
COPY inference ./inference
COPY models ./models

EXPOSE 8080

CMD ["uvicorn","inference.api:app","--host","0.0.0.0","--port","8080"]


