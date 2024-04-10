#!/usr/bin/env bash
set -euxo pipefail

if [ -z "${K_CMD:+''}" ]; then
    if command -v k3d &> /dev/null ; then
        K_CMD="k3d"
    elif command -v kind &> /dev/null ; then
        K_CMD="kind"
    else
        echo "Either k3d or kind should be available"
        echo "Refer to https://k3d.io/#releases"
        echo "Refer to https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
        exit 1
    fi
fi

if [ "$K_CMD" == "k3d" ]; then
  k3d cluster list | grep fluidos | awk '{print $1}' | xargs -I % k3d cluster delete %
elif command -v kind &> /dev/null ; then
  # TBD
  kind get clusters | grep fluidos | awk '{print $1}' | xargs -I % kind delete cluster --name %
fi
