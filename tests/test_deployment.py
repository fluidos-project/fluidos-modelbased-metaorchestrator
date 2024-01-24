import subprocess
import time
from kopf.testing import KopfRunner
import pytest
from pytest_kind import KindCluster
from pathlib import Path


kind_cluster: KindCluster = None


def setup_module(module):
    kind_cluster = KindCluster("foo-bar")
    kind_cluster.create()
    time.sleep(3)
    kind_cluster.kubectl("apply", "-f", Path(
        Path(__file__).parent.parent,
        "utils/fluidos-deployment-crd.yaml").absolute().as_posix())

    module.kind_cluster = kind_cluster


def teardown_module(module):
    module.kind_cluster.delete()


def test_operator_executes():
    with KopfRunner(["run", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        print(runner)
        pass

    assert runner.exit_code == 0


def test_get_fd_succeds():
    with KopfRunner(["run", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        assert not kind_cluster.kubectl("get", "fd")

    assert runner.exit_code == 0


def test_scheduling_successfull_single_pod():   #(kind_cluster: KindCluster):
    with KopfRunner(["run", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        assert not kind_cluster.kubectl("get", "fd")

    assert runner.exit_code == 0


# def test_scheduling_successfull_deployment_single_pod():
#     with KopfRunner(["run", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
#         subprocess.run("kubectl apply -f tests/examples/test-deployment.yaml", shell=True, check=True)
#         time.sleep(5)

#     assert runner.exit_code == 0
