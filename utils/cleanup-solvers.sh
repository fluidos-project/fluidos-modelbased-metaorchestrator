#!/usr/bin/env bash

# set -xeuo pipefail
set -euo pipefail

NAMESPACE=${1:-'fluidos'}

echo "Using namespace $NAMESPACE"

echo "Searching for fluidosdeployments ..."
for resource_name in $(kubectl get fd --no-headers -n $NAMESPACE 2> /dev/null | awk '{print $1}'); do
    echo "Removing fd/$resource_name"
    kubectl delete fd/$resource_name -n $NAMESPACE
done
echo "... done"

for resource_type in $(cat tests/node/crds/* | yq '.spec.names.plural' | grep -v -- '---' | grep -v flavors); do
    # kubectl get $resource_type --no-headers -n $NAMESPACE 2> /dev/null | awk '{print $1}' | xargs -I % kubectl delete $resource_type/% -n $NAMESPACE
    echo "Searching for $resource_type ..."
    for resource_name in $(kubectl get $resource_type --no-headers -n $NAMESPACE 2> /dev/null | awk '{print $1}'); do
        echo "Removing $resource_type/$resource_name"
        kubectl delete $resource_type/$resource_name -n $NAMESPACE
    done
    echo "... done"
done
