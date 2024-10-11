#!/usr/bin/env bash

set -xeuo pipefail

# function _get_command {
#   if command -v docker 2>/dev/null >&2 ; then
#     echo "docker"
#   else
#     echo "podman"
#   fi
# }


# $(_get_command) run --rm -it -e RABBITMQ_DEFAULT_USER=testuser -e RABBITMQ_DEFAULT_PASS=testuser123 -p 15672:15672 -p 5672:5672 rabbitmq:3-management

# setup testbed
./setup-testbed.sh

# create service flavor in the DE provider
kubectl apply -f $PWD/rabbitmq-service.yaml --kubeconfig $PWD/provider-DE-config.yaml
