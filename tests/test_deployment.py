import subprocess
import time
from kopf.testing import KopfRunner


def teardown_function(function):
    subprocess.run("kubectl get mbd | awk '{print $1}' | tail -n+2 | xargs kubectl delete mbd", shell=True, check=True)
    # subprocess.run("kubectl get pod | awk '{print $1}' | tail -n+2 | xargs kubectl delete pod ", shell=True, check=True)  # not required anymore
    subprocess.run("kubectl get solvers | awk '{print $1}' | tail -n+2 | xargs kubectl delete solvers", shell=True, check=True)
    subprocess.run("kubectl get peeringcandidates | awk '{print $1}' | tail -n+2 | xargs kubectl delete peeringcandidates", shell=True, check=True)


def setup_function(function):
    subprocess.run("kubectl get mbd | awk '{print $1}' | tail -n+2 | xargs kubectl delete mbd", shell=True, check=True)
    # subprocess.run("kubectl get pod | awk '{print $1}' | tail -n+2 | xargs kubectl delete pod ", shell=True, check=True)  # not required anymore
    subprocess.run("kubectl get solvers | awk '{print $1}' | tail -n+2 | xargs kubectl delete solvers", shell=True, check=True)
    subprocess.run("kubectl get peeringcandidates | awk '{print $1}' | tail -n+2 | xargs kubectl delete peeringcandidates", shell=True, check=True)


def test_operator_executes():
    with KopfRunner(["run", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        pass

    assert runner.exit_code == 0


def test_scheduling_successfull_single_pod():
    with KopfRunner(["run", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        subprocess.run("kubectl apply -f tests/examples/test-pod.yaml", shell=True, check=True)
        time.sleep(5)

    assert runner.exit_code == 0
    print(runner.output)


def test_scheduling_successfull_deployment_single_pod():
    with KopfRunner(["run", "--verbose", "-m", "fluidos_model_orchestrator"]) as runner:
        subprocess.run("kubectl apply -f tests/examples/test-deployment.yaml", shell=True, check=True)
        time.sleep(5)

    assert runner.exit_code == 0
