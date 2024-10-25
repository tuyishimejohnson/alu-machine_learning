#!/usr/bin/env python3
import requests

def availableShips(passengerCount):
    url = 'https://swapi.dev/api/starships/'
    ships = []
    
    while url:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            # Some ships may have unknown or non-numeric values in the passengers field, so we need to handle that
            try:
                passengers = ship['passengers'].replace(',', '')  # Remove any commas from the string
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except (ValueError, TypeError):
                continue
        
        url = data['next']  # Move to the next page if available
    
    return ships
