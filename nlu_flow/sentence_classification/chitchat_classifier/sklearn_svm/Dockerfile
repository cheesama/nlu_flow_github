FROM python:3.8-slim

USER root

COPY requirements.txt req.txt
RUN buildDeps='gcc libc6-dev make' \
    && apt-get update && apt-get install -y $buildDeps git wget tar \
    && pip install --upgrade pip \
    && pip install -r req.txt \
    && pip install git+http://github.com/cheesama/nlu_flow.git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove $buildDeps

WORKDIR /workspace/
COPY inferencer.py /workspace/inferencer.py
COPY chitchat_classifier_model.svc /workspace/chitchat_classifier_model.svc
COPY response_dict.dill /workspace/response_dict.dill

EXPOSE  8000
CMD ["uvicorn", "inferencer:app", "--reload", "--host=0.0.0.0", "--port=8000"]
