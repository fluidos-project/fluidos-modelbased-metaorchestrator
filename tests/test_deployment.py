from kopf.testing import KopfRunner
import pytest
from pytest_kubernetes.providers.base import AClusterManager
from pathlib import Path
from os import environ


@pytest.mark.xfail(environ.get("CI", "false") == "true", reason="Not running on Travis")
def test_successfull_submission(k8s: AClusterManager):
    k8s.create()
    k8s.apply(Path(Path(__file__).parent.parent, "utils/fluidos-deployment-crd.yaml"))
    with KopfRunner(["run", "-A", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        assert runner is not None
        assert 0 == len(k8s.kubectl(["get", "fd"])["items"])
        k8s.apply(Path(Path(__file__).parent, "examples/test-deployment.yaml"))
        names = [item['metadata']['name'] for item in k8s.kubectl(["get", "fd"])["items"]]
        assert 1 == len(names)
        for name in names:
            k8s.kubectl(f"delete fd {name}".split(), as_dict=False)
        assert 0 == len(k8s.kubectl(["get", "fd"])["items"])

    assert runner.exit_code == 0
