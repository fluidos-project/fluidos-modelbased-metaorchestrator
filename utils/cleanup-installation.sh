#!/usr/bin/env bash


set -euo pipefail

TARGET_NS=${1:-'default'}
FLUIDOS_NS=${2:-'fluidos'}

echo "Using namespace $TARGET_NS for deployment and $FLUIDOS_NS as Node namespace"

echo "Searching for FLUIDOSDeployments ..."
for resource_name in $(kubectl get fd --no-headers -n $TARGET_NS 2> /dev/null | awk '{print $1}'); do
    echo "Removing fd/$resource_name"

    kubectl delete fd/$resource_name -n $TARGET_NS
done
echo "... done"

echo "Removing namespaceoffloading resource"
for resource_name in $(kubectl get NamespaceOffloading --no-headers -n $TARGET_NS 2> /dev/null | awk '{print $1}'); do
    echo "Removing namespaceoffloading/$resource_name"

    kubectl delete namespaceoffloading/$resource_name -n $TARGET_NS
done


for resource_type in discoveries peeringcandidates allocations serviceblueprints solvers contracts reservations transactions ; do
    echo "Searching for $resource_type ..."
    for resource_name in $(kubectl get $resource_type --no-headers -n $FLUIDOS_NS 2> /dev/null | awk '{print $1}'); do
        echo "Removing $resource_type/$resource_name"
        kubectl delete $resource_type/$resource_name -n $FLUIDOS_NS
    done
    echo "... done"
done
