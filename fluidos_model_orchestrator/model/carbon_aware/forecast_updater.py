import logging

import requests
import datetime

from fluidos_model_orchestrator import ModelPredictRequest
from fluidos_model_orchestrator.resources import get_resource_finder
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')
HEADERS = {'auth-token': API_KEY}
BASE_URL = 'https://api.electricitymap.org/v3'

def _get_live_carbon_intensity(lat, lon):
    url = f"{BASE_URL}/carbon-intensity/latest"
    params = {'lat': lat, 'lon': lon}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()["carbonIntensity"]
    else:
        print(f"Error fetching live data: {response.status_code}")
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
        print(f"Error fetching forecasted data: {response.status_code}")
        print(f"Erro2: {response.reason}")
        return None


def update_local_flavours_forecasted_data(namespace: str):
    resources = get_resource_finder(None, None).retrieve_all_flavors(namespace)
    for resource in resources:
        lat = resource.location.get("latitude")
        lon = resource.location.get("longitude")
        new_forecast = _get_forecasted_carbon_intensity(lat, lon)
        new_forecast.insert(0, _get_live_carbon_intensity(lat, lon)) # index 0 = current intensity. Forecast starts at index 1
        new_forecast_timeslots = []
        for i in range(len(new_forecast) - 1):
            average = (new_forecast[i] + new_forecast[i + 1]) / 2
            new_forecast_timeslots.append(average)
        print("new_forecast from external API: ", new_forecast)
        print("new_forecast_timeslots: ", new_forecast_timeslots)
        get_resource_finder(None, None).update_local_flavor(resource, new_forecast_timeslots)
