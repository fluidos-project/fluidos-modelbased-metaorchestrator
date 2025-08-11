#!/usr/bin/env bash

set -euo pipefail


function _get_command {
  if command -v docker 2>/dev/null >&2 ; then
    echo "docker"
  else
    echo "podman"
  fi
}


# basic setup
helm repo add fluidos https://fluidos-project.github.io/node/ --force-update

CONSUMER_NODE_PORT=30000
PROVIDER_NODE_PORT=30001

COMMAND=$(_get_command)

# Mitigate issues related to resource usage on Kind, see https://github.com/fluidos-project/node/blob/main/docs/installation/installation.md
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  sudo swapoff -a
  sudo sysctl fs.inotify.max_user_instances=8192
  sudo sysctl fs.inotify.max_user_watches=524288
fi

# setup provider DE
kind create cluster --name provider-germany --config $PWD/provider-cluster-config.yaml --kubeconfig $PWD/provider-DE-config.yaml

DE_PROVIDER_CONTROLPLANE_IP=$($COMMAND inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' provider-germany-control-plane)

liqoctl install kind --kubeconfig $PWD/provider-DE-config.yaml

helm upgrade --install --devel -n fluidos --create-namespace node fluidos/node \
  --set "provider=kind" \
  --set "common.configMaps.nodeIdentity.ip=$DE_PROVIDER_CONTROLPLANE_IP" \
  --set "common.configMaps.nodeIdentity.domain=de.provider.fluidos.eu" \
  --set "common.configMaps.nodeIdentity.nodeId=provider-de" \
  --set "rearController.service.gateway.nodePort.port=$PROVIDER_NODE_PORT" \
  --set "networkManager.config.enableLocalDiscovery=false" \
  --wait \
  --kubeconfig $PWD/provider-DE-config.yaml

# Wait until at least one flavor resource is present
until kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-DE-config.yaml | grep -q .; do
  echo "Waiting for flavor resource to be created in provider-DE..."
  sleep 2
done

kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-DE-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ./flavors-location-germany.yaml --type merge -n fluidos --kubeconfig $PWD/provider-DE-config.yaml

# setup provider Italy
kind create cluster --name provider-italy --config $PWD/provider-cluster-config.yaml --kubeconfig $PWD/provider-IT-config.yaml

IT_PROVIDER_CONTROLPLANE_IP=$($COMMAND inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' provider-italy-control-plane)

liqoctl install kind --kubeconfig $PWD/provider-IT-config.yaml

helm upgrade --install --devel -n fluidos --create-namespace node fluidos/node \
  --set "provider=kind" \
  --set "common.configMaps.nodeIdentity.ip=$IT_PROVIDER_CONTROLPLANE_IP" \
  --set "common.configMaps.nodeIdentity.domain=it.provider.fluidos.eu" \
  --set "common.configMaps.nodeIdentity.nodeId=provider-it" \
  --set "rearController.service.gateway.nodePort.port=$PROVIDER_NODE_PORT" \
  --set "networkManager.config.enableLocalDiscovery=false" \
  --wait \
  --kubeconfig $PWD/provider-IT-config.yaml

# setup consumer
kind create cluster --name consumer --config $PWD/consumer-cluster-config.yaml --kubeconfig $PWD/consumer-config.yaml

CONSUMER_CONTROLPLANE_IP=$($COMMAND inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' consumer-control-plane)

liqoctl install kind --kubeconfig $PWD/consumer-config.yaml

helm upgrade --install --devel -n fluidos --create-namespace node fluidos/node \
  --set "provider=kind" \
  --set "common.configMaps.nodeIdentity.ip=$CONSUMER_CONTROLPLANE_IP" \
  --set "common.configMaps.nodeIdentity.domain=consumer.fluidos.eu" \
  --set "common.configMaps.nodeIdentity.nodeId=consumer" \
  --set "rearController.service.gateway.nodePort.port=$CONSUMER_NODE_PORT" \
  --set "networkManager.config.enableLocalDiscovery=false" \
  --wait \
  --kubeconfig $PWD/consumer-config.yaml

# add config map required to run MBMO
kubectl apply -f $PWD/example-mbmo-config-map.yaml --kubeconfig $PWD/consumer-config.yaml -n fluidos

# add CRDs required
kubectl apply --kubeconfig $PWD/consumer-config.yaml -f $PWD/../../deployment/fluidos-meta-orchestrator/crds

# Wait until at least one flavor resource is present
until kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-IT-config.yaml | grep -q .; do
  echo "Waiting for flavor resource to be created in provider-IT..."
  sleep 2
done
#
# Wait until at least one flavor resource is present
until kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/consumer-config.yaml | grep -q .; do
  echo "Waiting for flavor resource to be created in consumer..."
  sleep 2
done

# pretend the consumer cluster is in Dublin, Ireland
kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/consumer-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file $PWD/flavors-location-ireland.yaml --type merge -n fluidos --kubeconfig $PWD/consumer-config.yaml

# create KnowCluster resources in the consumer

cat <<-EOF | kubectl apply -f - --kubeconfig $PWD/consumer-config.yaml
apiVersion: network.fluidos.eu/v1alpha1
kind: KnownCluster
metadata:
  name: knowncluster-provider-de
  namespace: fluidos
spec:
  # Set ip:port with the provider cluster control plane
  address: ${DE_PROVIDER_CONTROLPLANE_IP}:${PROVIDER_NODE_PORT}
EOF

cat <<-EOF | kubectl apply -f - --kubeconfig $PWD/consumer-config.yaml
apiVersion: network.fluidos.eu/v1alpha1
kind: KnownCluster
metadata:
  name: knowncluster-provider-it
  namespace: fluidos
spec:
  # Set ip:port with the provider cluster control plane
  address: ${IT_PROVIDER_CONTROLPLANE_IP}:${PROVIDER_NODE_PORT}
EOF


if [ -n ${DEMO:-""} ]; then
  echo "Setting up demo environment"

  # configure IT to be AZURE
  kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-IT-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file $PWD/bandwidth-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-IT-config.yaml

  # configure TEE available in DE provider
  # kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-IT-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file ../../tests/examples/tee-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-IT-config.yaml
  kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-DE-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file $PWD/tee-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-DE-config.yaml

  # configure carbon emission in IT (bad)
  kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-IT-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file $PWD/carbon-good-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-IT-config.yaml

  # configure carbon emission in DE (good)
  kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-DE-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file $PWD/carbon-bad-patch-file.yaml --type merge -n fluidos --kubeconfig $PWD/provider-DE-config.yaml

  # configure DE as having sensors
  kubectl get flavor -n fluidos --no-headers --kubeconfig $PWD/provider-DE-config.yaml | cut -f1 -d\  | xargs -I% kubectl patch flavor/%  --patch-file $PWD/flavor-patch-sensor.yaml --type merge -n fluidos --kubeconfig $PWD/provider-DE-config.yaml

  # create service flavor in the IT provider
  kubectl apply -f $PWD/rabbitmq-service.yaml --kubeconfig $PWD/provider-IT-config.yaml
fi
