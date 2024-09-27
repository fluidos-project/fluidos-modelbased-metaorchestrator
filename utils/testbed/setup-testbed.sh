#!/usr/bin/env bash

set -xao pipefail


function _get_command {
  if command -v docker 2>/dev/null >&2 ; then
    return "docker"
  else
    return "podman"
  fi
}



# basic setup
helm repo add fluidos https://fluidos-project.github.io/node/ --force-update

consumer_node_port=30000
provider_node_port=30001

COMMAND=_get_command


# setup provider
kind create cluster --name provider --config $PWD/provider-cluster-config.yaml --kubeconfig $PWD/provider-config.yaml

provider_controlplane_ip=$($COMMAND inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' provider-control-plane)

liqoctl install kind --kubeconfig $PWD/provider-config.yaml

helm upgrade --install --devel -n fluidos --create-namespace node fluidos/node \
  --set "provider=kind" \
  --set "networkManager.configMaps.nodeIdentity.ip=$provider_controlplane_ip:$provider_node_port" \
  --set "networkManager.configMaps.nodeIdentity.domain=provider.fluidos.eu" \
  --set "networkManager.configMaps.nodeIdentity.nodeId=provider" \
  --wait \
  --kubeconfig $PWD/provider-config.yaml

# setup consumer
kind create cluster --name consumer --config $PWD/consumer-cluster-config.yaml --kubeconfig $PWD/consumer-config.yaml

consumer_controlplane_ip=$($COMMAND inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' consumer-control-plane)

liqoctl install kind --kubeconfig $PWD/consumer-config.yaml

helm upgrade --install --devel -n fluidos --create-namespace node fluidos/node \
  --set "provider=kind" \
  --set "networkManager.configMaps.nodeIdentity.ip=$consumer_controlplane_ip:$consumer_node_port" \
  --set "networkManager.configMaps.providers.local=$provider_controlplane_ip:$provider_node_port" \
  --set "networkManager.configMaps.nodeIdentity.domain=consumer.fluidos.eu" \
  --set "networkManager.configMaps.nodeIdentity.nodeId=consumer" \
  --wait \
  --kubeconfig $PWD/consumer-config.yaml

kubectl apply -f $PWD/example-mbmo-config-map.yaml --kubeconfig $PWD/consumer-config.yaml


# pretend the consumer cluster is in Dublin, Ireland
kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/consumer-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ./flavors-location.yaml --type merge -n fluidos --kubeconfig $PWD/consumer-config.yaml
