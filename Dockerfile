FROM python:3.11

WORKDIR /app

ADD fluidos_model_orchestrator/ /app/fluidos_model_orchestrator
ADD setup.cfg setup.py /app/

RUN pip install .

CMD kopf run -m fluidos_model_orchestrator --verbose