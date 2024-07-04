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

    # Get live and forecasted carbon intensity
    live_data = get_live_carbon_intensity(lon, lat)
    forecasted_data = get_forecasted_carbon_intensity(lon, lat)
    carbon_intensity_values = []
    if live_data:
        carbon_intensity_values.append(live_data['carbonIntensity'])
        if forecasted_data:
            print("\nForecasted Carbon Intensity:")
            print(forecasted_data)
            #get all carbonIntensity values and store them in a list
            #JSON structure: {'zone': 'US-CAL-CISO', 'forecast': [{'carbonIntensity': 173, 'datetime': '2024-07-03T14:00:00.000Z'}, {'carbonIntensity': 144, 'datetime': '2024-07-03T15:00:00.000Z'}, {'carbonIntensity': 127, 'datetime': '2024-07-03T16:00:00.000Z'}, {'carbonIntensity': 123, 'datetime': '2024-07-03T17:00:00.000Z'}, {'carbonIntensity': 123, 'datetime': '2024-07-03T18:00:00.000Z'}, {'carbonIntensity': 125, 'datetime': '2024-07-03T19:00:00.000Z'}, {'carbonIntensity': 130, 'datetime': '2024-07-03T20:00:00.000Z'}, {'carbonIntensity': 141, 'datetime': '2024-07-03T21:00:00.000Z'}, {'carbonIntensity': 168, 'datetime': '2024-07-03T22:00:00.000Z'}, {'carbonIntensity': 176, 'datetime': '2024-07-03T23:00:00.000Z'}, {'carbonIntensity': 193, 'datetime': '2024-07-04T00:00:00.000Z'}, {'carbonIntensity': 229, 'datetime': '2024-07-04T01:00:00.000Z'}, {'carbonIntensity': 286, 'datetime': '2024-07-04T02:00:00.000Z'}, {'carbonIntensity': 314, 'datetime': '2024-07-04T03:00:00.000Z'}, {'carbonIntensity': 320, 'datetime': '2024-07-04T04:00:00.000Z'}, {'carbonIntensity': 321, 'datetime': '2024-07-04T05:00:00.000Z'}, {'carbonIntensity': 328, 'datetime': '2024-07-04T06:00:00.000Z'}, {'carbonIntensity': 322, 'datetime': '2024-07-04T07:00:00.000Z'}, {'carbonIntensity': 306, 'datetime': '2024-07-04T08:00:00.000Z'}, {'carbonIntensity': 302, 'datetime': '2024-07-04T09:00:00.000Z'}, {'carbonIntensity': 300, 'datetime': '2024-07-04T10:00:00.000Z'}, {'carbonIntensity': 303, 'datetime': '2024-07-04T11:00:00.000Z'}, {'carbonIntensity': 309, 'datetime': '2024-07-04T12:00:00.000Z'}, {'carbonIntensity': 292, 'datetime': '2024-07-04T13:00:00.000Z'}, {'carbonIntensity': 173, 'datetime': '2024-07-04T14:00:00.000Z'}, {'carbonIntensity': 144, 'datetime': '2024-07-04T15:00:00.000Z'}, {'carbonIntensity': 127, 'datetime': '2024-07-04T16:00:00.000Z'}, {'carbonIntensity': 123, 'datetime': '2024-07-04T17:00:00.000Z'}, {'carbonIntensity': 123, 'datetime': '2024-07-04T18:00:00.000Z'}, {'carbonIntensity': 125, 'datetime': '2024-07-04T19:00:00.000Z'}, {'carbonIntensity': 130, 'datetime': '2024-07-04T20:00:00.000Z'}, {'carbonIntensity': 141, 'datetime': '2024-07-04T21:00:00.000Z'}, {'carbonIntensity': 168, 'datetime': '2024-07-04T22:00:00.000Z'}, {'carbonIntensity': 176, 'datetime': '2024-07-04T23:00:00.000Z'}, {'carbonIntensity': 193, 'datetime': '2024-07-05T00:00:00.000Z'}, {'carbonIntensity': 229, 'datetime': '2024-07-05T01:00:00.000Z'}, {'carbonIntensity': 286, 'datetime': '2024-07-05T02:00:00.000Z'}, {'carbonIntensity': 314, 'datetime': '2024-07-05T03:00:00.000Z'}, {'carbonIntensity': 320, 'datetime': '2024-07-05T04:00:00.000Z'}, {'carbonIntensity': 321, 'datetime': '2024-07-05T05:00:00.000Z'}, {'carbonIntensity': 328, 'datetime': '2024-07-05T06:00:00.000Z'}, {'carbonIntensity': 322, 'datetime': '2024-07-05T07:00:00.000Z'}, {'carbonIntensity': 306, 'datetime': '2024-07-05T08:00:00.000Z'}, {'carbonIntensity': 302, 'datetime': '2024-07-05T09:00:00.000Z'}, {'carbonIntensity': 300, 'datetime': '2024-07-05T10:00:00.000Z'}, {'carbonIntensity': 303, 'datetime': '2024-07-05T11:00:00.000Z'}, {'carbonIntensity': 309, 'datetime': '2024-07-05T12:00:00.000Z'}, {'carbonIntensity': 292, 'datetime': '2024-07-05T13:00:00.000Z'}], 'updatedAt': '2024-07-03T13:53:57.169Z'}

            for forecast in forecasted_data['forecast']:
                carbon_intensity_values.append(forecast['carbonIntensity'])
            print(carbon_intensity_values)
