import random


def generate_electricity_maps_forecast(deadline: int) -> list[int]:
    return [
        random.randint(10, 1_000) for _ in range(deadline)
    ]
