import logging

import requests

from fluidos_model_orchestrator.resources import get_resource_finder
from fluidos_model_orchestrator.configuration import Configuration
from fluidos_model_orchestrator.common import Flavor

config = Configuration()
API_KEY = config.api_keys.get("ELECTRICITY_MAP_API_KEY")
HEADERS = {'auth-token': API_KEY}
BASE_URL = 'https://api.electricitymap.org/v3'

def _get_live_carbon_intensity(lat, lon):
    url = f"{BASE_URL}/carbon-intensity/latest"
    params = {'lat': lat, 'lon': lon}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()["carbonIntensity"]
    else:
        logging.debug(f"Error fetching live data: {response.status_code}")
        return None


def _get_forecasted_carbon_intensity(lat, lon):
    url = f"{BASE_URL}/carbon-intensity/forecast"
    params = {'lat': lat, 'lon': lon}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        forecast_values = []
        for forecast_item in response.json()["forecast"]:
            forecast_values.append(forecast_item['carbonIntensity'])
        return forecast_values
    else:
        logging.debug(f"Error fetching forecasted data: {response.status_code}")
        logging.debug(f"Error: {response.reason}")
        return None


def update_local_flavor_forecasted_data(flavor: Flavor, namespace: str) -> None:
    lat = flavor.location.get("latitude")
    lon = flavor.location.get("longitude")
    new_forecast = _get_forecasted_carbon_intensity(lat, lon)
    new_forecast.insert(0,
                        _get_live_carbon_intensity(lat, lon))  # index 0 = current intensity. Forecast starts at index 1
    new_forecast_timeslots = []
    for i in range(len(new_forecast) - 1):
        average = (new_forecast[i] + new_forecast[i + 1]) / 2
        new_forecast_timeslots.append(average)
    logging.debug("new_forecast from external API: ", new_forecast)
    logging.debug("new_forecast_timeslots: ", new_forecast_timeslots)
    optionalField = {"operational": new_forecast_timeslots}
    get_resource_finder(None, None).update_local_flavor(flavor, optionalField, namespace)
