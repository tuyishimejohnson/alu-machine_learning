#!/usr/bin/env python3
"""
Script that prints the location of a specific user using the GitHub API.
"""

import requests
import time
from datetime import datetime
import sys

def main(url):
    """
    - The user is passed as first argument of the script
    with the full API URL, example: ./2-user_location.py
    https://api.github.com/users/holbertonschool
    - If the user doesn’t exist, print Not found
    - If the status code is 403, print Reset in X min where X
    is the number of minutes from now and the value of
    X-Ratelimit-Reset
    - Your code should not be executed when the file is
    imported (you should use if __name__ == '__main__':)
    """
    response = requests.get(url)
    
    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_time = int(response.headers.get('X-Ratelimit-Reset'))
        current_time = int(time.time())
        minutes_to_reset = (reset_time - current_time) // 60
        print(f"Reset in {minutes_to_reset} min")
    else:
        user_data = response.json()
        location = user_data.get('location', 'Location not available')
        print(location)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
    else:
        main(sys.argv[1])
