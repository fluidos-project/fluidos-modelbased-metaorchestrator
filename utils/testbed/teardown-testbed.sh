#!/usr/bin/env bash

set -xauo pipefail

kind get clusters | xargs kind delete clusters
