#!/usr/bin/env bash
set -xeu

CWD=`dirname $0`
CWD=`realpath $CWD`

WORKER_LABEL="fluidos.eu/resource-node"

if [ ! -d $CWD/logs ]; then
    mkdir $CWD/logs
fi

echo "milan is the provider"
kind create cluster --config $CWD/cluster-multi-worker.yaml --name milan --kubeconfig $CWD/examples/milan-kubeconfig.yaml

export KUBECONFIG=$CWD/examples/milan-kubeconfig.yaml

kubectl get nodes

kubectl apply -f node/deployments/fluidos/crds
kubectl label nodes milan-worker $WORKER_LABEL=true

kubectl apply -f $CWD/metrics-server.yaml

kubectl -n kube-system wait --for=condition=ready pod -l k8s-app=metrics-server --timeout 300s

pushd node
go run cmd/local-resource-manager/main.go  > $CWD/logs/milan-lrm.out 2> $CWD/logs/milan-lrm.err &
echo "$!" > $CWD/logs/milan-lrm.pid
go run cmd/rear-controller/main.go  --http-port :14145 > $CWD/logs/milan-rc.out 2> $CWD/logs/milan-rc.err &
echo "$!" > $CWD/logs/milan-rm.pid
popd

echo "dublin is the consumer"

kind create cluster --config $CWD/cluster-multi-worker.yaml --name dublin --kubeconfig $CWD/examples/dublin-kubeconfig.yaml

export KUBECONFIG=$CWD/examples/dublin-kubeconfig.yaml

kubectl get nodes

kubectl apply -f node/deployments/fluidos/crds
kubectl label nodes dublin-worker $WORKER_LABEL=true

kubectl apply -f $CWD/metrics-server.yaml

kubectl -n kube-system wait --for=condition=ready pod -l k8s-app=metrics-server --timeout 300s

pushd node
go run cmd/local-resource-manager/main.go  > $CWD/logs/dublin-lrm.out 2> $CWD/logs/dublin-lrm.err &
echo "$!" > $CWD/logs/dublin-lrm.pid
go run cmd/rear-controller/main.go  --http-port :14146 --metrics-bind-address :9082 --health-probe-bind-address :9083 --server-address http://localhost:14145/api > $CWD/logs/dublin-rc.out 2> $CWD/logs/dublin-rc.err &
echo "$!" > $CWD/logs/dublin-rc.pid
go run cmd/rear-manager/main.go > $CWD/logs/dublin-rm.out 2> $CWD/logs/dublin-rm.err &
echo "$!" > $CWD/logs/dublin-rm.pid
popd
