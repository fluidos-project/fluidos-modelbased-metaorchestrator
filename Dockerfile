FROM python:3.12.11

ENV NAMESPACE=fluidos

WORKDIR /app

RUN apt-get update
RUN apt-get install -y libhdf5-serial-dev
ADD fluidos_model_orchestrator/ /app/fluidos_model_orchestrator
ADD setup.cfg setup.py /app/

RUN pip install .

# required by baseline model
ENV TOKENIZERS_PARALLELISM=false

CMD kopf run -n ${NAMESPACE} -m fluidos_model_orchestrator --verbose
