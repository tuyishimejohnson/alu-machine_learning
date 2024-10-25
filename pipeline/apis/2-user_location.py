#!/usr/bin/env python3
"""
Script to print the location of a GitHub user.
"""

import requests
import sys
from datetime import datetime

def get_user_location(url):
    """
    Retrieves the location of a specified GitHub user.
    """
    response = requests.get(url)
    
    if response.status_code == 200:
        # Success: retrieve and return location if available
        data = response.json()
        return data.get('location', "Not found")
    
    elif response.status_code == 404:
        # User not found
        return "Not found"
    
    elif response.status_code == 403:
        # Rate limit exceeded
        reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
        reset_in_minutes = (datetime.fromtimestamp(reset_time) - datetime.now()).total_seconds() // 60
        return f"Reset in {int(reset_in_minutes)} min"

    # For other errors, return a generic message
    return "Error retrieving data"


if __name__ == '__main__':
    # Check that the URL argument is provided
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API user URL>")
        sys.exit(1)
    
    url = sys.argv[1]
    print(get_user_location(url))
