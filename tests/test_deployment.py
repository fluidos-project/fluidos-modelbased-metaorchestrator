# import subprocess
import time
from kopf.testing import KopfRunner
import pytest
from pytest_kind import KindCluster
from pathlib import Path


# kind_cluster: KindCluster = None
#
#
# def setup_module(module):
#     kind_cluster = KindCluster("foo-bar")
#     kind_cluster.create()
#     module.kind_cluster = kind_cluster
# def teardown_module(module):
#     module.kind_cluster.delete()
#     module.kind_cluster = None


@pytest.mark.skip
def test_operator_executes(kind_cluster: KindCluster):
    time.sleep(3)
    kind_cluster.kubectl("apply", "-f", Path(
        Path(__file__).parent.parent,
        "utils/fluidos-deployment-crd.yaml").absolute().as_posix())
    time.sleep(3)
    with KopfRunner(["run", "-A", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        print(runner)
        pass

    assert runner.exit_code == 0


@pytest.mark.skip
def test_get_fd_succeds(kind_cluster: KindCluster):
    time.sleep(3)
    kind_cluster.kubectl("apply", "-f", Path(
        Path(__file__).parent.parent,
        "utils/fluidos-deployment-crd.yaml").absolute().as_posix())
    time.sleep(3)
    with KopfRunner(["run", "-A", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        assert not kind_cluster.kubectl("get", "fd")

    assert runner.exit_code == 0


@pytest.mark.skip
def test_scheduling_successfull_single_pod(kind_cluster: KindCluster):
    time.sleep(3)
    kind_cluster.kubectl("apply", "-f", Path(
        Path(__file__).parent.parent,
        "utils/fluidos-deployment-crd.yaml").absolute().as_posix())
    time.sleep(3)
    with KopfRunner(["run", "-A", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        assert not kind_cluster.kubectl("apply", "-f", "tests/examples/test-deployment.yaml")
        time.sleep(5)

    assert runner.exit_code == 0
