from typing import cast

import pytest  # type: ignore
import requests_mock  # type: ignore

from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.intent import KnownIntent
from fluidos_model_orchestrator.common.prometheus import has_intent_validation_failed
from fluidos_model_orchestrator.common.prometheus import retrieve_metric


@pytest.mark.skip()
def test_against_real() -> None:
    metrics = retrieve_metric(
        "latency", "http://localhost:9090"
    )

    assert metrics is not None
    assert False


def test_empty_response() -> None:
    with requests_mock.Mocker() as m:
        mocker = cast(requests_mock.Mocker, m)  # TMMYH

        mocker.register_uri(method="GET", url="http://dummy-hostname.org:9090/api/v1/query", json={
            "status": "success", "data": {"resultType": "matrix", "result": []}
        })

        metrics = retrieve_metric(
            "latency", "http://dummy-hostname.org:9090"
        )

        assert metrics is not None


def test_not_empty_response() -> None:
    with requests_mock.Mocker() as m:
        mocker = cast(requests_mock.Mocker, m)  # TMMYH

        mocker.register_uri(method="GET", url="http://dummy-hostname.org:9090/api/v1/query", json={
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": []
            }
        })

        metrics = retrieve_metric(
            "gc_call", "http://dummy-hostname.org:9090"
        )

        assert metrics is not None


def test_intent_validation_no_data() -> None:
    with requests_mock.Mocker() as m:
        mocker = cast(requests_mock.Mocker, m)  # TMMYH

        mocker.register_uri(method="GET", url="http://dummy-hostname.org:9090/api/v1/query", json={
            "status": "success", "data": {"resultType": "matrix", "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {
                            "__name__": "fluidos_latency",
                            "cluster": "cluster",
                            "exported_job": "cluster1",
                            "instance": "45654a7fd718.ngrok-free.app",
                            "job": "testing"
                        },
                        "values": [
                            [
                                1754907768.270,
                                "100"
                            ],
                            [
                                1754907770.270,
                                "100"
                            ],
                            [
                                1754907772.270,
                                "100"
                            ]
                        ]
                    }
                ]
            }}
        })

        assert not has_intent_validation_failed(
            intent=Intent(name=KnownIntent.latency, value="200"),  # 200ms latency
            prometheus_ref="http://dummy-hostname.org:9090",
            domain="provider.fluidos.eu",
            namespace="my namespace",
            name="myname"
        )


def test_intent_validation_good_data() -> None:
    with requests_mock.Mocker() as m:
        mocker = cast(requests_mock.Mocker, m)  # TMMYH

        mocker.register_uri(method="GET", url="http://dummy-hostname.org:9090/api/v1/query", json={
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {
                            "__name__": "fluidos_latency",
                            "cluster": "cluster",
                            "exported_job": "cluster1",
                            "instance": "45654a7fd718.ngrok-free.app",
                            "job": "testing"
                        },
                        "values": [
                            [
                                1754907630.270,
                                "100"
                            ],
                            [
                                1754907632.270,
                                "100"
                            ],
                            [
                                1754907634.270,
                                "200"
                            ]
                        ]
                    }
                ]
            }
        })
        assert not has_intent_validation_failed(
            intent=Intent(name=KnownIntent.latency, value="200"),  # 200ms latency
            prometheus_ref="http://dummy-hostname.org:9090",
            domain="",
            namespace="my namespace",
            name="myname"
        )


def test_intent_validation_bad_data() -> None:
    with requests_mock.Mocker() as m:
        mocker = cast(requests_mock.Mocker, m)  # TMMYH

        mocker.register_uri(method="GET", url="http://dummy-hostname.org:9090/api/v1/query", json={
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {
                            "__name__": "fluidos_latency",
                            "cluster": "cluster",
                            "exported_job": "cluster1",
                            "instance": "45654a7fd718.ngrok-free.app",
                            "job": "testing"
                        },
                        "values": [
                            [
                                1754907630.270,
                                "500"
                            ],
                            [
                                1754907632.270,
                                "500"
                            ],
                            [
                                1754907634.270,
                                "500"
                            ]
                        ]
                    }
                ]
            }
        })
        assert has_intent_validation_failed(
            intent=Intent(name=KnownIntent.latency, value="200"),  # 200ms latency
            prometheus_ref="http://dummy-hostname.org:9090",
            domain="provider.fluidos.eu",
            namespace="my namespace",
            name="myname"
        )


def test_intent_robot_status() -> None:
    with requests_mock.Mocker() as m:
        mocker = cast(requests_mock.Mocker, m)  # TMMYH

        mocker.register_uri(method="GET", url="http://dummy-hostname.org:9090/api/v1/query", json={
            "status": "success", "data": {"resultType": "matrix", "result": []}
        })

        response = has_intent_validation_failed(
            intent=Intent(name=KnownIntent.latency, value="200"),  # 200ms latency
            prometheus_ref="http://dummy-hostname.org:9090",
            domain="provider.fluidos.eu",
            namespace="my namespace",
            name="myname"
        )

        assert not response
