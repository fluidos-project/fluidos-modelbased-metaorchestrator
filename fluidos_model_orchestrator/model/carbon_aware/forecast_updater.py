import logging
from typing import cast

import requests

from fluidos_model_orchestrator.common.flavor import Flavor
from fluidos_model_orchestrator.common.flavor import FlavorK8SliceData
from fluidos_model_orchestrator.common.flavor import FlavorType
from fluidos_model_orchestrator.configuration import CONFIGURATION


def _get_live_carbon_intensity(lat: str, lon: str) -> int:
    BASE_URL = 'https://api.electricitymap.org/v3'
    API_KEY = CONFIGURATION.api_keys['ELECTRICITY_MAP_API_KEY']
    HEADERS = {'auth-token': str(API_KEY)}
    url = f"{BASE_URL}/carbon-intensity/latest"
    params = {'lat': lat, 'lon': lon}

    logging.debug(f"Request URL: {url}")
    logging.debug(f"Request params: {params}")

    response = requests.get(url, headers=HEADERS, params=params)

    logging.debug(f"Response status code: {response.status_code}")
    logging.debug(f"Response content: {response.content!r}")

    if response.status_code == 200:
        return response.json()["carbonIntensity"]
    else:
        logging.exception(f"Error fetching live data: {response.status_code} - {response.text}")
        raise RuntimeError("Error fatching data")


def _get_forecasted_carbon_intensity(lat: str, lon: str) -> list[int] | None:
    BASE_URL = 'https://api.electricitymap.org/v3'
    API_KEY = CONFIGURATION.api_keys['ELECTRICITY_MAP_API_KEY']
    HEADERS = {'auth-token': str(API_KEY)}
    url = f"{BASE_URL}/carbon-intensity/forecast"
    params = {'lat': lat, 'lon': lon}
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        forecast_values: list[int] = []
        for forecast_item in response.json()["forecast"]:
            forecast_values.append(forecast_item['carbonIntensity'])
        return forecast_values
    else:
        logging.exception(f"Error fetching forecasted data: {response.status_code}")
        logging.exception(f"Error: {response.reason}")
    return None


def update_local_flavor_forecasted_data(flavor: Flavor, namespace: str) -> Flavor | None:
    flavor_type_enum = flavor.spec.flavor_type.type_identifier
    if flavor_type_enum is not FlavorType.K8SLICE:
        logging.error("Flavor type is not K8S")
        return None

    location = flavor.spec.location

    if location is None:
        logging.error("Flavor location is None")
        return flavor

    if "latitude" not in location or "longitude" not in location:
        logging.error("Flavor location does not contain latitude or longitude")
        return None

    lat = str(flavor.spec.location.get("latitude"))
    lon = str(flavor.spec.location.get("longitude"))

    logging.debug(f"Updating flavor's forecasted carbon emission for {lat=}, {lon=}")

    new_forecast = _get_forecasted_carbon_intensity(lat, lon)

    if new_forecast is None:
        logging.exception("Unable to retrieve forecast")
        return None

    new_forecast.insert(0, _get_live_carbon_intensity(lat, lon))  # index 0 = current intensity. Forecast starts at index 1

    new_forecast_timeslots = []
    for i in range(len(new_forecast) - 1):
        average = (new_forecast[i] + new_forecast[i + 1]) / 2
        new_forecast_timeslots.append(average)
        new_forecast_timeslots_int = [int(x) for x in new_forecast_timeslots]

    logging.debug("new_forecast from external API: %s", new_forecast)
    logging.debug("new_forecast_timeslots: %s", new_forecast_timeslots_int)

    k8sSlice_data: FlavorK8SliceData = cast(FlavorK8SliceData, flavor.spec.flavor_type.type_data)

    k8sSlice_data.properties["carbon-footprint"] = {
        "embodied": k8sSlice_data.properties["carbon-footprint"].get("embodied", 0),
        "operational": new_forecast_timeslots_int
    }

    return flavor
