FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.railway.txt .
RUN pip install --no-cache-dir -r requirements.railway.txt

COPY . .

ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

EXPOSE 8001

CMD ["python", "run.py"]