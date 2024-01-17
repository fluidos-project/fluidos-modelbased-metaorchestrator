from kubernetes import config
from kubernetes import client


def test_retrieve_local_flavous():
    config.load_config()
    api = client.CustomObjectsApi()

    response = api.list_cluster_custom_object(group="nodecore.fluidos.eu", version="v1alpha1", plural="flavours")

    assert response
    assert len(response)
    assert len(response["items"])
    assert response["items"][0]["kind"] == "Flavour"
    assert response["items"][0]["spec"]
    assert len(response["items"][0]["spec"])
    assert response["items"][0]["spec"]["type"] == "k8s-fluidos"


# def test_solver():
#     config.load_config()
#     api = client.CustomObjectsApi()
#     response = api.create_namespaced_custom_object(group="nodecore.fluidos.eu", version="v1alpha1", namespace="defalt", plural="flavours", body={
#         "kind": "Solver"
#     })
