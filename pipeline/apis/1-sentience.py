#!/usr/bin/env python3

"""
A method that returns the list of names
of the home planets of all sentient species.
"""


import requests

def sentientPlanets():
    """
    Arguments:
    passengerCount: number of passengers

    Returns:
    If no ship available, return an empty list.
    """
    url_link = 'https://swapi.dev/api/species/'
    planets = set()

    while url_link:
        response = requests.get(url_link)

        if response.status_code != 200:
            print(f"Failed to retrieve data: {response.status_code}")
            break

        data = response.json()

        for species in data['results']:

            if species['classification'] == 'sentient':
                homeworld_url = species['homeworld']

                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    
                    if homeworld_response.status_code == 200:
                        homeworld_data = homeworld_response.json()
                        planets.add(homeworld_data['name'])
                    else:
                        print(f"Failed to retrieve homeworld data for {species['name']}")
                else:
                    planets.add("unknown")
        url_link = data['next']

    return list(planets)
