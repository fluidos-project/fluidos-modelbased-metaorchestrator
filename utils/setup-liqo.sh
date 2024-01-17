#!/usr/bin/env bash

set -xeu

CWD=`dirname $0`
CWD=`realpath $CWD`

liqoctl install kind --cluster-name dublin --kubeconfig $CWD/examples/dublin-kubeconfig.yaml
liqoctl install kind --cluster-name milan --kubeconfig $CWD/examples/milan-kubeconfig.yaml

liqoctl offload namespace default --kubeconfig $CWD/examples/dublin-kubeconfig.yaml
