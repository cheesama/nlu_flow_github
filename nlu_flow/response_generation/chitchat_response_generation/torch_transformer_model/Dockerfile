FROM python:3.8-slim

USER root

COPY requirements.txt req.txt
RUN buildDeps='gcc libc6-dev make' \
    && apt-get update && apt-get install -y $buildDeps git wget tar \
    && pip install --upgrade pip \
    && pip install -r req.txt \
    && pip install git+http://github.com/cheesama/nlu_flow.git \
    && pip install git+http://github.com/cheesama/soynlp.git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove $buildDeps

WORKDIR /workspace/
COPY embedding_transformer.py /workspace/embedding_transformer.py
COPY transformer_chitchat_response_model.modeldict /workspace/transformer_chitchat_response_model.modeldict
COPY inferencer.py /workspace/inferencer.py

EXPOSE  8000
CMD ["uvicorn", "inferencer:app", "--reload", "--host=0.0.0.0", "--port=8000"]
