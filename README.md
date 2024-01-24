# fluidos-model-orchestrator

This repository contains the operator to perform model-based meta-orchestration within a Cloud-to-Edge continuum as described by FLUIDOS.

## Requirements

The operator assumes the following:
* Kubernetes version >= 28.1.0
* REAR (node) functionality version >= *MISSING*

Moreover, the interaction with the operator assumes:
* fluidos-kubectl-plugin version >= *MISSING*

To run the operator in development mode, the following is required:
* python >= 3.11

## How to run the operator

The operator can be executed in two main modes, namely development mode and production mode.
The former refers to the the operator being executed within a local environment against a running kubernetes cluster (usually Kind). The latter, on the other hand, refers to the operator running within a kubernetes cluster.

### Development mode

### Production mode


## Examples


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