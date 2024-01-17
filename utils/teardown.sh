#!/usr/bin/env bash
set -eu

for _ in `seq 3`; do
  ps aux | grep go- | grep -v grep | awk '{print $2}' | xargs kill -9
done

kind get clusters | xargs kind delete clusters
