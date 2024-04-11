<!-- markdownlint-disable first-line-h1 -->
<p align="center">
<a href="https://www.fluidos.eu/"> <img src="./docs/images/fluidoslogo.png" width="150"/> </a>
<h3 align="center">FLUIDOS Model-based Meta-Orchestrator</h3>
</p>

This repository contains the operator to perform model-based meta-orchestration within a Cloud-to-Edge continuum as described by FLUIDOS.

## Requirements

The operator assumes the following:
* Kubernetes version >= 28.1.0
* REAR (as implemented in [node](https://github.com/fluidos-project/node)) functionality version >= 0.0.3
* Liqo version >= 0.9.4

Moreover, the interaction with the operator assumes:
* [kubectl-fluidos-plugin](https://github.com/fluidos-project/kubectl-fluidos-plugin) version >= 0.0.2

To run the operator in development mode, the following is required:
* python >= 3.11

## How to run the operator

The operator can be executed in two main modes, namely development mode and production mode.
The former refers to the the operator being executed within a local environment against a running kubernetes cluster (usually Kind). The latter, on the other hand, refers to the operator running within a kubernetes cluster.

### Development mode

Development mode assumes access to a Kubernetes cluster. An example of cluster, using kind is available [here](utils/cluster-multi-worker.yaml).

```bash
# start kind
kind create cluster --name foo --config utils/cluster-multi-worker.yaml --kubeconfig utils/examples/dublin-kubeconfig.yaml

# install CRD
kubectl apply -f utils/fluidos-deployment-crd.yaml

# start FLUIDOS operator
kopfs run --verbose -m fluidos_model_orchestrator
```

The shell will provide the log of the execution of the operator.

### Production mode

When deploying directly on a cluster, one can leverage the following utility steps:

```bash
# build docker image
docker build -t fluidos-mbmo:latest . && docker push

# install CRD
kubectl apply -f utils/fluidos-deployment-crd.yaml

# install operator to cluster
kubectl apply -f utils/fluidos-deployment.yaml
```

Note that the docker image must be available to the cluster. If the cluster has been created with kind, the image must be loaded using `kind load docker-image fluidos-mbmo:latest`. Also, note that if the environment is using podman instead of docker, then alternative steps are required. Namely, the docker image must be loaded into the cluster nodes via the following steps:
`podman save fluidos-mbmo:latest -o /tmp/fluidos-mbmo-latest.tar && kind load image-archive /tmp/fluidos-mbmo-latest.tar`.

### Example of interaction

TODO

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md)
for details on our process for submitting pull requests to us, and please ensure
you follow the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

To install the environment for the local development,
read [DEVELOPMENT.md](DEVELOPMENT.md).


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [releases on this repository](https://github.com/fluidos-project/fluidos-modelbased-metaorchestrator/releases).


## License

This project is licensed under the Apache License — version 2.0
see the [LICENSE](LICENSE) file for details.
