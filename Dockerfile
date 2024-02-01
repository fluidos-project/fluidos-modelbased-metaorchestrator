FROM python:3.11

WORKDIR /app

ADD fluidos_model_orchestrator/ /app/fluidos_model_orchestrator
ADD setup.cfg setup.py /app/

RUN pip install .

# required by baseline model
ENV TOKENIZERS_PARALLELISM=false

CMD kopf run -A -m fluidos_model_orchestrator --verbose