#!/usr/bin/env bash

get_cluster_ip_and_port() {
    case "$1" in
    "kind")
        echo $(kubectl config --kubeconfig utils/examples/$2-kubeconfig.yaml view | yq '.clusters[0].cluster.server' | cut -f3 -d/)
        ;;
    "k3d")
        echo $(k3d kubeconfig get $2 | yq '.clusters[0].cluster.server' | cut -f3 -d/)
        ;;
    *)
        echo "cluster manager unset"
        exit 1
        ;;
    esac
}

build_cluster() {
    CWD=$(realpath $(dirname $0))

    case "$1" in
    "kind")
        kind create cluster --config $CWD/cluster-multi-worker.yaml --name $2 --kubeconfig $CWD/examples/$2-kubeconfig.yaml --wait 20s
        docker images | grep fluidos | awk '{print $1":"$2}' | xargs -I % kind load docker-image % --name=$2
        liqoctl install kind -n liqo --kubeconfig $CWD/examples/$2-kubeconfig.yaml --cluster kind-$2
        kubectl apply -f "$CWD/metrics-server.yaml" --kubeconfig $CWD/examples/$2-kubeconfig.yaml || true  # because somethimes kubectl complains about the metrics-server file, but it works..
        echo "Waiting for metrics-server to be ready"
        kubectl wait --for=condition=ready pod -l k8s-app=metrics-server -n kube-system --timeout=300s --kubeconfig $CWD/examples/$2-kubeconfig.yaml
        ;;
    "k3d")
        k3d cluster create $2 --agents $(cat $CWD/cluster-multi-worker.yaml | grep 'role: worker' | wc -l) --wait
        kubectl label nodes -l '!node-role.kubernetes.io/control-plane' node-role.fluidos.eu/resources=true
        docker images | grep fluidos | awk '{print $1":"$2}' | xargs -I % k3d image import % -c $2
        liqoctl install k3s -n liqo --cluster k3d-$2
        kubectl apply -f "$CWD/metrics-server.yaml" || true  # because somethimes kubectl complains about the metrics-server file, but it works..
        echo "Waiting for metrics-server to be ready"
        kubectl wait --for=condition=ready pod -l k8s-app=metrics-server -n kube-system --timeout=300s
        ;;
    *)
        echo "cluster manager unset"
        exit 1
        ;;
    esac
}

install_fluidos_node() {
    CWD=$(realpath $(dirname $0))

    case "$1" in
    "kind")
        helm install --namespace fluidos fluidos-node $CWD/../node/deployments/node --values $CWD/fluidos-node-values.yaml \
            --set networkManager.configMaps.nodeIdentity.ip="$3" \
            --set networkManager.configMaps.providers.local="$4" \
            --create-namespace \
            --kubeconfig $CWD/examples/$2-kubeconfig.yaml \
            --wait
        ;;
    "k3d")
        kubectl config use-context k3d-$2
        helm install --namespace fluidos fluidos-node $CWD/../node/deployments/node --values $CWD/fluidos-node-values.yaml \
            --set networkManager.configMaps.nodeIdentity.ip="$3" \
            --set networkManager.configMaps.providers.local="$4" \
            --create-namespace \
            --wait
        ;;
    *)
        echo "cluster manager unset"
        exit 1
        ;;
    esac
}
