import json
import logging
from time import sleep
from typing import Any

import requests

from fluidos_model_orchestrator.common.intent import Intent
from fluidos_model_orchestrator.common.resource import ResourceProvider
from fluidos_model_orchestrator.configuration import CONFIGURATION


logger = logging.getLogger(__name__)


def request_application(policy: str, endpoint: str, request_name: str) -> str:
    endpoint = endpoint + "/" + request_name
    try:
        response = requests.post(endpoint, data=policy.strip(), headers={
            "Content-Type": "application/xml"
        })

        get_endpoint: str | None = None

        for _ in range(CONFIGURATION.n_try):
            if response.status_code // 100 == 1:
                sleep(CONFIGURATION.API_SLEEP_TIME)
                if get_endpoint is None:
                    get_endpoint = response.headers.get("Location", endpoint)
                response = requests.get(get_endpoint)
            if response.status_code // 100 == 1:
                sleep(CONFIGURATION.API_SLEEP_TIME)
                if get_endpoint is None:
                    get_endpoint = response.headers.get("Location", endpoint)
                response = requests.get(get_endpoint)

            if response.status_code // 100 == 2:
                return response.text
            if response.status_code // 100 == 4:
                raise RuntimeError("Error on our side")
            if response.status_code // 100 == 5:
                raise RuntimeError("Error on their side")

    except Exception as e:
        logger.error("Unable to perform requeest")
        logger.error(e)

    raise RuntimeError("Unable to receive a response in the required time")


def create_mspl(provider: str, consumer: str, exporterEndpoint: str, properties: dict[str, Any]) -> str:
    # exported endpoint is the URL to the proxy fluidos: http://otel-prometheus-proxy.proxy-prom-test.svc.cluster.local:8080/api/v1/write

    # metrics_xml = '\n'.join(f'<nameMetric>{value}</nameMetric>' for value in properties.values())
    metrics_xml = json.dumps(
        [{
            "filter/sendSpecificMetric": {
                "error_mode": "ignore",
                "metrics": {
                    "metric": [
                        "not (\n  IsMatch(name, \"^node.fluidos.*\") or\n  resource.attributes[\"container.id\"] == \"testing\"\n)"
                    ]
                }
            }
        }]
    )

    xml = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
                <ITResourceOrchestration id="omspl_46bdc9a9035540d4b257bd686a7e6bc3"
                    xmlns="http://modeliosoft/xsddesigner/a22bd60b-ee3d-425c-8618-beb6a854051a/ITResource.xsd"
                    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                    xsi:schemaLocation="http://modeliosoft/xsddesigner/a22bd60b-ee3d-425c-8618-beb6a854051a/ITResource.xsd mspl.xsd">
                    <ITResource id="mspl_46bdc9a9035540d4b257bd686a7e6c55" orchestrationID="omspl_46bdc9a9035540d4b257bd686a7e6bc5">
                        <configuration xsi:type="RuleSetConfiguration">
                            <capability>
                                <Name>Telemetry</Name>
                            </capability>
                            <configurationRule>
                                <configurationRuleAction xsi:type="TelemetryAction">
                                <telemetryActionType>TRANSFER</telemetryActionType>
                                </configurationRuleAction>
                                <configurationCondition xsi:type='TransferConfigurationConditions'>
                                    <isCNF>false</isCNF>
                                    <transferConfigurationCondition>
                                        <domainID>{provider}</domainID>
                                        <flavorID>{consumer}</flavorID>
                                        <exporterEndpoint>{exporterEndpoint}</exporterEndpoint>
                                        <properties>
                                            {metrics_xml}
                                        </properties>
                                    </transferConfigurationCondition>
                                </configurationCondition>
                                <Description>Transfer metrics from cluster UMU (flavorID {consumer}) to IBM</Description>
                                <Name>Transfer01</Name>
                                <isCNF>false</isCNF>
                            </configurationRule>
                            <Name>Conf1</Name>
                        </configuration>
                        <priority>1000</priority>
                    </ITResource>
            </ITResourceOrchestration>"""
    return xml


def request_telemetry_for(intents: list[Intent], provider: ResourceProvider) -> bool:
    logger.warning("Not implemented: fix!!!")
    return True
