import requests
import datetime

API_KEY = 'REbFju22LwHPVT3t1Y0IKh1I'
HEADERS = {'auth-token': API_KEY}

# Base URL for electricityMap API
BASE_URL = 'https://api.electricitymap.org/v3'

# Function to get live carbon intensity
def get_live_carbon_intensity(lon, lat):
    url = f"{BASE_URL}/carbon-intensity/latest"
    params = {'lon': lon, 'lat': lat}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching live data: {response.status_code}")
        return None

# Function to get forecasted carbon intensity
def get_forecasted_carbon_intensity(lon, lat):
    url = f"{BASE_URL}/carbon-intensity/forecast"
    params = {'lon': lon, 'lat': lat}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching forecasted data: {response.status_code}")
        return None

# Example usage
if __name__ == "__main__":
    # Coordinates for San Francisco
    lon = -122.4194
    lat = 37.7749

    # Get live carbon intensity
    live_data = get_live_carbon_intensity(lon, lat)
    if live_data:
        print("Live Carbon Intensity:")
        print(live_data)

    # Get forecasted carbon intensity
    forecasted_data = get_forecasted_carbon_intensity(lon, lat)
    if forecasted_data:
        print("\nForecasted Carbon Intensity:")
        for forecast in forecasted_data['forecast']:
            time = datetime.datetime.fromisoformat(forecast['datetime'])
            carbon_intensity = forecast['carbonIntensity']
            print(f"Time: {time}, Carbon Intensity: {carbon_intensity} gCO2eq/kWh")