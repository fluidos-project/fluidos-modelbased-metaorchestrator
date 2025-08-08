# Testbed

This folder contains scripts and data for the setup of a demo scenario.

This scenario involves three FLUIDOS nodes:
* Two providers, virtually located one in Berlin and one in Turin
* One consumer, virtually located in Dublin.

The basic installation, obtained by using the [setup-testbed.sh](setup-testbed.sh) script creates the basic for such scenario.
It is also possible to add the following characterstics by running as follows:
```bash
DEMO=1 ./setup-testbed.sh
```

Setting the `DEMO` environment variable the script will also configure the following additional characteristics in the providers:

- Germany:
    - TEE support available
    - Good carbon emission values
    - Availability of sensors (humidity and CO2 detectors)
- Italy:
    - Configure it to be executed in AZURE
    - Bad carbon emission values
    - Support for service flavor (rabbitMQ)


We can execute the metaorchestrator. We remind that the metaorchestrator is a python project, which requires its dependencies to be installed.
This installation can be obtained as follows.

First let us create a virtualenviroment:
```bash
python -m venv venv
source ./venv/bin/activate
```

We then install the metaorchestrator and its dependencies. This assumes that the current working directory is `utils/testbed`, the `cd` command can be removed if we are alredy in the root folder of the project.
```bash
cd ../..
pip install .
```

It is now possible to run the metaorchestrator, instructing it to connecto the consumer cluster, by setting the `KUBECONFIG` environment varible to the correct configuration file.

```bash
export KUBECONFIG=$PWD/utils/testbed/consumer-config.yaml

kopf run -m fluidos_model_orchestrator -A
```

In another shell we can then submit one of the example files and observe the behavior of the testbed.
As an example, let us deploy a simple scenario in which the user requires to deploy a workload with a certain latency requirement (say 100ms) and the availability of a humidity sensor. This is represented in this [file](../demo/task-1.2-latency-sensor.yaml):

```yaml
apiVersion: fluidos.eu/v1
kind: FLUIDOSDeployment
metadata:
  name: y2demo-task-1.2
spec:
  apiVersion: v1
  kind: Pod
  metadata:
    name: y2demo-task-1.2
    annotations:
      fluidos-intent-latency: "100"
      fluidos-intent-sensor: "humidity"
  spec:
    containers:
    - image: nginx:latest
      imagePullPolicy: IfNotPresent
      name: producer
      resources:
        requests:
          memory: "64Mi"
          cpu: "250m"
        limits:
          memory: "128Mi"
          cpu: "500m"
```

We can deploy this as follwos:

```bash
export KUBECONFIG=$PWD/utils/testbed/consumer-config.yaml

kubectl create ns demo-ns
kubectl apply -n demo-ns -f utils/demo/task-1.2-latency-sensor.yaml
```
