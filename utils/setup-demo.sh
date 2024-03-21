#!/usr/bin/env bash

set -euxo pipefail

CWD=`dirname $0`
CWD=`realpath $CWD`

build_cluster () {
    case "$1" in

    "kind")
        kind create cluster --config $CWD/cluster-multi-worker.yaml --name $2 --kubeconfig $CWD/examples/$2-kubeconfig.yaml
        docker images | grep fluidos | awk '{print $1":"$2}' | xargs -I % kind load docker-image % --name=$2
        liqoctl install kind -n liqo --kubeconfig $CWD/examples/$2-kubeconfig.yaml --cluster $2
        ;;
    "k3d")
        k3d cluster create $2 --agents $(cat $CWD/cluster-multi-worker.yaml | grep 'role: worker' | wc -l) --wait
        kubectl get nodes -l '!node-role.kubernetes.io/control-plane' --no-headers | awk '{print $1}' | xargs -I % kubectl label nodes % node-role.fluidos.eu/resources=true
        docker images | grep fluidos | awk '{print $1":"$2}' | xargs -I % k3d image import % -c $2
        liqoctl install k3s -n liqo --cluster k3d-$2
        ;;
    *)
        echo "cluster manager unset"
        exit 1
        ;;
    esac

    # kubectl apply -f "$PWD/metrics-server.yaml"

    # echo "Waiting for metrics-server to be ready"
    # kubectl wait --for=condition=ready pod -l k8s-app=metrics-server -n kube-system --timeout=300s   

    helm install --namespace fluidos fluidos-node $CWD/../node/deployments/node --values $CWD/fluidos-node-values.yaml --set "networkManager.configMaps.nodeIdentity.domain=$2.at.ibm.fluidos.eu" --create-namespace
}

K_CMD=""

if ! command -v k3d &2>/dev/null ; then
    K_CMD="k3d"
elif ! command -v kind &2>/dev/null ; then
    K_CMD="kind"
fi

pushd ../node/tools/development
for x in $(cat build.sh | grep VALID_COMPONENTS= | cut -f2 -d\( | tr -d ')"' ); do
    bash build.sh fluidos test $x
done
popd

build_cluster $K_CMD fluidos-provider
build_cluster $K_CMD fluidos-consumer


