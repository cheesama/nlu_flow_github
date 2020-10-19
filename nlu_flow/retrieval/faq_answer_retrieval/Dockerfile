FROM python:3.7-slim

USER root

COPY requirements.txt req.txt
RUN buildDeps='gcc libc6-dev make libomp-dev' \
    && apt-get update && apt-get install -y $buildDeps git wget tar \
    && pip install --upgrade pip \
    && pip install -r req.txt \
    && pip install faiss --no-cache \
    && pip install git+http://github.com/cheesama/nlu_flow.git \
    && pip install git+http://github.com/cheesama/soynlp.git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove $buildDeps

WORKDIR /workspace/
COPY koelectra_fine_tuner.py /workspace/koelectra_fine_tuner.py
COPY koelectra_faq_retrieval_model.modeldict /workspace/koelectra_faq_retrieval_model.modeldict
COPY faq_retrieval_index /workspace/faq_retrieval_index
COPY inferencer.py /workspace/inferencer.py

EXPOSE  8000
CMD ["uvicorn", "inferencer:app", "--reload", "--host=0.0.0.0", "--port=8000"]
