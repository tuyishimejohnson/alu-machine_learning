#!/usr/bin/env python3

"""
A method that returns the list of ships that can
hold a given number of passengers
"""
import requests


def availableShips(passengerCount):

    """
    Arguments:
    passengerCount: number of passengers

    Returns:
    If no ship available, return an empty list.
    """

    url = 'https://swapi.dev/api/starships/'
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            try:
                passengers = ship['passengers'].replace(',', '')    
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except (ValueError, TypeError):
                continue

        url = data['next']  # Move to the next page if available

    return ships