#!/usr/bin/env bash
set -xeuo pipefail

CWD=`realpath $(dirname $0)`

source $CWD/functions.sh


if ! command -v liqoctl &> /dev/null ; then
    echo "liqoctl not found. Refer to https://docs.liqo.io/en/v0.9.4/installation/liqoctl.html"
    exit 1
fi

if ! command -v jq &> /dev/null ; then
    echo "jq not found. Refer to https://jqlang.github.io/jq/"
    exit 1
fi

if ! command -v yq &> /dev/null ; then
    echo "yq not found. Refer to https://github.com/mikefarah/yq"
    exit 1
fi

if [ -z "${K_CMD+''}" ]; then
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

if [ ! -d node ]; then
    git clone git@github.com:fluidos-project/node.git
else
    pushd node
    git checkout main
    git pull origin main
    popd
fi

if [ ! "${SKIP_BUILD_IMAGES:-''}" == "true" ]; then
    pushd node/tools/development
    NAMESPACE=fluidos
    VERSION=test
    for COMPONENT in $(cat build.sh | grep VALID_COMPONENTS= | cut -f2 -d\( | tr -d ')"' ); do
        docker buildx build -f ../../build/common/Dockerfile --build-arg COMPONENT="$COMPONENT" -t "$NAMESPACE"/"$COMPONENT":"$VERSION" ../../
    done
    popd
fi

build_cluster $K_CMD fluidos-provider
build_cluster $K_CMD fluidos-consumer

provider_ip_and_port=$(get_cluster_ip_and_port $K_CMD fluidos-provider)
consumer_ip_and_port=$(get_cluster_ip_and_port $K_CMD fluidos-consumer)

install_fluidos_node $K_CMD fluidos-provider $provider_ip_and_port $consumer_ip_and_port
install_fluidos_node $K_CMD fluidos-consumer $consumer_ip_and_port $provider_ip_and_port
