import subprocess
from kubernetes import config, client, utils
import yaml
import pkg_resources


def setup_function(function):
    subprocess.run("kubectl get pod | awk '{print $1}' | tail -n+2 | xargs kubectl delete pod ", shell=True, check=True)  # not required anymore


def test_loading_of_objects():
    myconfig = client.Configuration()
    config.kube_config.load_kube_config(client_configuration=myconfig)

    k8s_client = client.ApiClient(myconfig)

    with pkg_resources.resource_stream(__name__, "k8s/pod.yaml") as pod_stream:
        pod_dict = yaml.safe_load(pod_stream)

    assert pod_dict

    print(f"{pod_dict=}")

    pod = utils.create_from_dict(k8s_client=k8s_client, data=pod_dict)

    assert pod is not None
