#!/usr/bin/env bash

set -euo pipefail

kind get clusters | xargs kind delete clusters

rm -f provider-{DE,IT}-config.yaml
rm -f consumer-config.yaml
