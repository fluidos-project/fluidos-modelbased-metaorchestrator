#!/usr/bin/env bash
set -eux

if ! command -v liqoctl &> /dev/null ; then
    echo "liqoctl not found. Refer to https://docs.liqo.io/en/v0.9.4/installation/liqoctl.html"
    exit 1
fi

if ! command -v kind &> /dev/null ; then
    echo "kind not found. Refer to https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    exit 1
fi

if [ ! -d node ]; then
    git clone git@github.com:fluidos-project/node.git
fi
pushd node
git checkout c9bfdd9
popd

CWD=`dirname $0`
CWD=`realpath $CWD`

$CWD/setup-rear.sh

$CWD/setup-liqo.sh

kubectl --kubeconfig $CWD/examples/dublin-kubeconfig.yaml apply -f $CWD/model-based-deployment-crd.yaml
