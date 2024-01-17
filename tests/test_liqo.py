import subprocess
from fluidos_model_orchestrator.resources import _extract_command, _extract_remote_cluster_id


def test_extract_command():
    # this test assumes kind cluster milan to be operating
    response = subprocess.run("liqoctl generate peer-command --kubeconfig \"$PWD/utils/examples/milan-kubeconfig.yaml\"", shell=True, check=True, capture_output=True)

    command = _extract_command(response.stdout)

    known_command_structure = "liqoctl peer out-of-band milan --auth-url SECRET --cluster-id SECRET --auth-token SECRET"

    assert all(
        part == known for part, known in zip(
            command.split(),
            known_command_structure.split()
        ) if known != "SECRET"
    )


def test_extract_remote_cluster_id():
    # this test assumes kind cluster milan to be operating
    template_command = "liqoctl peer out-of-band milan --auth-url SECRET --cluster-id CLUSTER_ID --auth-token SECRET"

    assert _extract_remote_cluster_id(template_command) == "CLUSTER_ID"
