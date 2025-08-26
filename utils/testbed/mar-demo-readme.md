
# The Martel Smart City Demo on FLUIDOS

A tiny “sensor” app, a model-based meta-orchestrator, and a three-cluster edge cloud—this post walks through how to run Martel’s Smart City demo on the **FLUIDOS** infrastructure from a fresh machine to a working `/sensor` endpoint. Along the way, you’ll see how **Flavors** describe capabilities (e.g., *which provider has temperature vs. humidity sensors*), and how the **Meta-Orchestrator (MBMO)** places workloads accordingly.

> **What you’ll build**
> A Python-based microservice that exposes `GET /sensor` returning JSON with `temperature` and `humidity`, deployed on a consumer cluster that can discover and reason about providers (Italy & Germany) via FLUIDOS.

---

## Why FLUIDOS?

FLUIDOS turns heterogeneous infrastructure (edge, fog, cloud, etc.) into a **market of resources**. Apps express *intents* (latency, locality, features like “has humidity sensors”), and FLUIDOS matches those to **Flavors** (capability descriptions) and **KnownClusters**. The **Model-Based Meta-Orchestrator** (MBMO) consumes your app spec, enriches it, and deploys it to the right place.

---

## Architecture at a glance

```
+--------------------+      +--------------------+      +--------------------+
|  Provider: Germany |      |  Provider: Italy   |      |     Consumer       |
|  (humidity sensor) |      | (temperature sens.)|      | (runs MBMO + app)  |
|  Flavor: sensors=[h]|      | Flavor: sensors=[t]|     |                    |
+----------+---------+      +----------+---------+      +----------+---------+
           \                         /                            |
            \                       /                             |
             +------ discovery ----+                    FLUIDOSDeployment
                       via FLUIDOS                      -> Pod + Service
```

* **Providers** advertise capabilities through **Flavors** (we’ll patch these to say “humidity” in DE, “temperature” in IT).
* **Consumer** runs the MBMO locally with `kopf`, sees your FD, and creates the sensor Pod and Service.

---

## Prerequisites

* macOS or Linux with:

  * `docker` (or `podman`), `kind`, `kubectl`, `helm`, `liqoctl`, `python3` + `venv`
* A few GB of disk/RAM free (Torch & embeddings are used by MBMO).
* Internet access for pulling images and Python packages.


---


# Deployment Instructions

## A) Download and Bring-up (Terminal 1)

```bash
git clone https://github.com/amjadmajid/fluidos-modelbased-metaorchestrator
cd fluidos-modelbased-metaorchestrator
```

**1) Create the 3-cluster testbed**

```bash
cd utils/testbed && bash setup-testbed.sh && cd -
```

**2) Export kubeconfigs**

```bash
export KCFG_DE="$PWD/utils/testbed/provider-DE-config.yaml"
export KCFG_IT="$PWD/utils/testbed/provider-IT-config.yaml"
export KCFG_C="$PWD/utils/testbed/consumer-config.yaml"
export KUBECONFIG="$KCFG_C"
kubectl config current-context   # should be: kind-consumer
kubectl cluster-info
```

**3) Patch provider Flavors**

```bash
# Germany → humidity
kubectl --kubeconfig "$KCFG_DE" -n fluidos \
  get flavor --no-headers | awk '{print $1}' | \
  xargs -I% kubectl --kubeconfig "$KCFG_DE" -n fluidos \
    patch flavor/% --type merge --patch-file utils/testbed/flavor-patch-humidity.yaml

# Italy → temperature
kubectl --kubeconfig "$KCFG_IT" -n fluidos \
  get flavor --no-headers | awk '{print $1}' | \
  xargs -I% kubectl --kubeconfig "$KCFG_IT" -n fluidos \
    patch flavor/% --type merge --patch-file utils/testbed/flavor-patch-temperature.yaml

# quick check you should see ['humidity'] and  ['temperature']
F_DE=$(kubectl --kubeconfig "$KCFG_DE" -n fluidos get flavor -o name | head -n1)
kubectl --kubeconfig "$KCFG_DE" -n fluidos get "$F_DE" \
  -o jsonpath='{.spec.flavorType.typeData.properties.additionalProperties.sensors}'; echo

F_IT=$(kubectl --kubeconfig "$KCFG_IT" -n fluidos get flavor -o name | head -n1)
kubectl --kubeconfig "$KCFG_IT" -n fluidos get "$F_IT" \
  -o jsonpath='{.spec.flavorType.typeData.properties.additionalProperties.sensors}'; echo
```

**4) Start the operator**

```bash
# from repo root
python -m venv venv
source venv/bin/activate

# (only if you changed code)
pip uninstall -y fluidos-model-orchestrator 2>/dev/null || true
rm -rf build dist *.egg-info
pip install -e .

# Make sure operator targets the consumer cluster
kopf run -m fluidos_model_orchestrator -A
```

Leave it running.

---

## B) Deploy the demo (Terminal 2)


**1) Apply the demo**

```bash
# point kubectl at consumer in THIS terminal
export KUBECONFIG="$PWD/utils/testbed/consumer-config.yaml"

# apply service+namespace then FD
kubectl apply -f utils/testbed/mar-demo-ns-svc.yaml
kubectl apply -f utils/testbed/mar-demo-fd.yaml

# watch the pod appear
kubectl -n sensors get pods -w
# or:
kubectl -n sensors wait --for=condition=Ready pod/temp-sensor --timeout=120s

# verify endpoints
kubectl -n sensors get po,svc,endpoints -o wide
```

**2) Test**

```bash
kubectl -n sensors port-forward svc/temp-sensor 8080:8080
```

**3) In a new terminal (Terminal 3), test the sensor endpoint**
```bash
# point kubectl at consumer in THIS terminal
export KUBECONFIG="$PWD/utils/testbed/consumer-config.yaml"
curl -s http://localhost:8080/sensor
```

---


## C) Teardown (delete everything)

**1) Stop the operator**

* In Terminal 1 (where `kopf` is running), press `Ctrl-C`.

**2) Teardown script (drop all KIND clusters & kubeconfig files)**

Run it from repo root:

```bash
cd utils/testbed/ && bash teardown-testbed.sh && cd -
```

**3) Clear local MBMO image/cache (optional but helpful)**

```bash
docker image rm -f local/fluidos-mbmo:dev 2>/dev/null || true
docker system prune -f                    2>/dev/null || true
```

**4) Prevent “localhost:8080” / bad kubeconfig errors**

```bash
unset KUBECONFIG KCFG_C KCFG_DE KCFG_IT
```







