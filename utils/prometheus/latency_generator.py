import logging
import random
import sys
import time

from prometheus_client import CollectorRegistry  # type: ignore
from prometheus_client import Gauge  # type: ignore
from prometheus_client import push_to_gateway  # type: ignore


logger = logging.getLogger(__name__)


class MetricGenerator:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        #------// Declare Gauges for Prometheus client
        self.latency = Gauge("latency", "Latency between current node and entrypoint", registry=self.registry)
        self.throughput = Gauge("throuput", "Application throughput (req/sec)", registry=self.registry)
        self.bandwidth_to_endpoint = Gauge("bandwidth-to-A", "Bandwidth available against node A", registry=self.registry)

    def update_gauges(self) -> None:
        self.latency.set(
            200 * random.random()
        )
        self.throughput.set(
            200 * random.random()
        )
        self.bandwidth_to_endpoint.set(
            500 * random.random()
        )

    def push(self, where: str, provider: str, application: str) -> None:
        push_to_gateway(where, job="fluidos", registry=self.registry, grouping_key={
            "provider": provider,
            "application": application,
        })


def main() -> int:
    logger.info("Performance Monitor Node Started")

    pushgateway_address = sys.argv[1]
    provider = sys.argv[2]
    application = sys.argv[3]

    generator = MetricGenerator()

    while True:
        sleep = 3
        print(f"{sleep=}")
        time.sleep(sleep)
        generator.update_gauges()
        generator.push(pushgateway_address, provider, application)


if __name__ == '__main__':
    raise SystemExit(main())
