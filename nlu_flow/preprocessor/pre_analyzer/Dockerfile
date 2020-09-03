FROM python:3.8-slim

USER root

RUN buildDeps='gcc libc6-dev make' \
    && apt-get update && apt-get install -y $buildDeps git wget tar \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove $buildDeps

RUN pip install git+http://github.com/cheesama/nlu_flow.git

WORKDIR /workspace/
COPY inferencer.py /workspace/inferencer.py

EXPOSE  8000
CMD ["uvicorn", "inferencer:app", "--reload", "--host=0.0.0.0", "--port=8000"]
