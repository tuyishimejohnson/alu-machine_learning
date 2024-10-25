#!/usr/bin/env python3
"""
    script that prints the location
    of a specific user:
"""


import requests
import sys
from datetime import datetime


def get_user_location(api_url):
    """
    Fetch and print the location of a GitHub user.

    :param api_url: The API URL for the user
    """
    response = requests.get(url)

    if response.status_code == 200:
        # Successfully retrieved data; return location if available
        data = response.json()
        return data.get('location', "Not found")
    
    elif response.status_code == 404:
        # User not found
        return "Not found"
    
    elif response.status_code == 403:
        # Rate limit exceeded
        reset_timestamp = int(response.headers.get('X-Ratelimit-Reset', 0))
        reset_in_minutes = (datetime.fromtimestamp(reset_timestamp) - datetime.now()).total_seconds() // 60
        return f"Reset in {int(reset_in_minutes)} min"

    # For other errors, return a generic message
    return "Error retrieving data"


if __name__ == '__main__':
    # Ensure that the URL argument is provided
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API user URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    print(get_user_location(url))