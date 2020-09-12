FROM python:3.8-slim

USER root

COPY requirements.txt req.txt
RUN buildDeps='gcc libc6-dev make libgomp1' \
    && apt-get update && apt-get install -y $buildDeps git wget tar \
    && pip install --upgrade pip \
    && pip install -r req.txt \
    && pip install git+http://github.com/SKT-AI/KoGPT2.git \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /workspace/
COPY inferencer.py /workspace/inferencer.py

EXPOSE  8000
CMD ["uvicorn", "inferencer:app", "--reload", "--host=0.0.0.0", "--port=8000"]
