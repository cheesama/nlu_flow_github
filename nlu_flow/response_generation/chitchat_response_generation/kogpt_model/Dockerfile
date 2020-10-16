FROM python:3.8-slim

USER root

COPY requirements.txt req.txt
RUN buildDeps='gcc libc6-dev make libgomp1' \
    && apt-get update && apt-get install -y $buildDeps git wget tar \
    && pip install --upgrade pip \
    && pip install --upgrade psutil \
    && pip install -r req.txt \
    && pip install git+http://github.com/SKT-AI/KoGPT2 \
    && pip install git+http://github.com/cheesama/soynlp.git \
    && pip install git+http://github.com/cheesama/nlu_flow.git \
    && rm -rf /var/lib/apt/lists/* 

WORKDIR /workspace/
COPY KoGPT2-chatbot/model_chp/model_last.ckpt /workspace/model_chp/model_-last.ckpt
COPY KoGPT2-chatbot/Chatbot_data /workspace/Chatbot_data
COPY KoGPT2-chatbot/inferencer.py /workspace/inferencer.py
COPY KoGPT2-chatbot/train_torch.py /workspace/train_torch.py

EXPOSE  8000
CMD ["python", "inferencer.py"]
