import logging
import random
import sys
import time

from prometheus_client import CollectorRegistry  # type: ignore
from prometheus_client import Gauge  # type: ignore
from prometheus_client import push_to_gateway  # type: ignore


logger = logging.getLogger(__name__)


class BatteryMonitor:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        #------// Declare Gauges for Prometheus client
        self.voltage_gauge = Gauge("battery_voltage", "Voltage of the battery in volts", registry=self.registry)
        self.current_gauge = Gauge("battery_current", "Current drawn by the battery in amperes", registry=self.registry)
        self.level_gauge = Gauge("battery_level", "Battery level as a percentage", registry=self.registry)
        self.time_remaining_gauge = Gauge("battery_time_remaining", "Estimated time remaining on battery in minutes", registry=self.registry)
        self.time_charging_gauge = Gauge("battery_time_charging", "Time spent charging the battery in minutes", registry=self.registry)
        self.is_charging_gauge = Gauge("battery_status", "Charging status of the battery (1 for true, 0 for false)", registry=self.registry)

    def update_gauges(self) -> None:
        #------// On set the metrics are updated on the exposed port

        self.voltage_gauge.set(round(random.uniform(3.0, 4.2), 2))
        self.current_gauge.set(round(random.uniform(0.5, 2.0), 2))
        self.level_gauge.set(random.randint(0, 100))
        self.time_remaining_gauge.set(random.randint(0, 120))
        self.time_charging_gauge.set(random.randint(0, 60))
        self.is_charging_gauge.set(random.choice([0, 1]))

    def push(self, where: str, job: str) -> None:
        push_to_gateway(where, job=job, registry=self.registry, grouping_key={
            "provider": job,
            "application": "something_else"
        })


def main() -> int:
    logger.info("Battery Monitor Node Started")

    pushgateway_address = sys.argv[1]
    robot_id = sys.argv[2]

    battery_monitor = BatteryMonitor()

    while True:
        sleep = 3
        print(f"{sleep=}")
        time.sleep(sleep)
        battery_monitor.update_gauges()
        battery_monitor.push(pushgateway_address, robot_id)  # localhost:9091", "robot-xyz")


if __name__ == '__main__':
    raise SystemExit(main())
