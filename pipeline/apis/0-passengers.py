#!/usr/bin/env python3
"""
Module to retrieve a list of starships that can hold a given number of passengers
from the SWAPI (Star Wars API). The method availableShips handles pagination,
error handling, and filtering of starships based on the given passenger count.
"""

import requests


def availableShips(passengerCount):
    """
    Method to return a list of starships that can accommodate at least
    a specified number of passengers. It interacts with the SWAPI (Star Wars API),
    fetches starships data, handles pagination, and filters based on passenger count.
    
    Args:
        passengerCount (int): Minimum number of passengers that the starship should be able to hold.
        
    Returns:
        list: A list of starship names that can hold the given number of passengers.
        If no starship meets the criteria, an empty list is returned.
    """
    # Base URL for the starships endpoint
    url_link = 'https://swapi-api.alx-tools.com/api/starships/'
    
    # List to store ships that match the criteria
    list_ships = []
    
    # Loop through paginated data as long as there are more pages
    while url_link:
        # Send a GET request to the current page URL
        response = requests.get(url_link)
        
        # Error handling: check if the request was successful (status code 200)
        if response.status_code != 200:
            print(f"Failed to retrieve data: {response.status_code}")
            break
        
        # Parse the JSON response
        data = response.json()

        # Iterate through each starship in the results
        for ship in data['results']:
            # Skip ships with invalid or unknown passenger values
            if ship["passengers"] not in ["n/a", "unknown", "0", "none"]:
                try:
                    # Remove commas from passenger numbers (if any) and convert to integer
                    ship["passengers"] = ship["passengers"].replace(",", "")
                    
                    # Check if the starship can hold the required number of passengers
                    if int(ship['passengers']) >= passengerCount:
                        # If it can, add the ship's name to the list
                        list_ships.append(ship['name'])
                except ValueError:
                    # Handle any errors in converting the passenger value to an integer
                    # (e.g., unexpected non-numeric values)
                    pass

        # Check if there is another page of results, and if so, continue to the next page
        url_link = data['next']
    
    # Return the final list of ships that meet the passenger criteria
    return list_ships
