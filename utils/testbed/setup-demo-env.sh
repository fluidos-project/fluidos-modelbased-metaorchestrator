#!/usr/bin/env bash

set -euo pipefail

date

./setup-testbed.sh

# create service flavor in the DE provider
# kubectl apply -f $PWD/rabbitmq-service.yaml --kubeconfig $PWD/provider-DE-config.yaml

# configure IT to be AZURE
# kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-IT-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ../../tests/examples/bandwidth-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-IT-config.yaml

# configure TEE available for both providers
# kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-IT-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ../../tests/examples/tee-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-IT-config.yaml
# kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-DE-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ../../tests/examples/tee-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-DE-config.yaml

# configure carbon emission in IT (bad)
kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-IT-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ../../tests/examples/carbon-good-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-IT-config.yaml

# configure carbon emission in DE (good)
kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-DE-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ../../tests/examples/carbon-bad-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-DE-config.yaml


# create namespaces
kubectl create ns my-ns --kubeconfig $PWD/consumer-config.yaml

date
