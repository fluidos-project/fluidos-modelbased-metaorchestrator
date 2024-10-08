from requests_mock import Mocker  # type: ignore

from fluidos_model_orchestrator.resources.mspl import request_application


endpoint = "http://www.um.es/mspl/endpoint"


def test_request_no_poll(requests_mock: Mocker) -> None:
    requests_mock.post(endpoint, status_code=200, text="response")

    text = request_application("<stupid><xml /></stupid>", endpoint)

    assert text == "response"


def test_request_poll(requests_mock: Mocker) -> None:
    requests_mock.post(endpoint, status_code=100, headers={'Location': endpoint + "123"})
    requests_mock.get(endpoint + "123", status_code=200, text="response")

    text = request_application("<stupid><xml /></stupid>", endpoint)

    assert text == "response"
