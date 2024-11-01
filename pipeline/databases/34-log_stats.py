#!/usr/bin/env python3
"""
This script Displays Nginx logs stored in MongoDB
"""
from pymongo import MongoClient

def log_stats():
    # Connect to MongoDB server (assuming it's running locally on default port)
    client = MongoClient('mongodb://localhost:27017/')
    
    # Access the 'logs' database and 'nginx' collection
    db = client.logs
    nginx_collection = db.nginx

    # 1. Count all logs in the collection
    total_logs = nginx_collection.count_documents({})
    print(f"{total_logs} logs")

    # 2. Count by HTTP methods
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print("Methods:")
    for method in methods:
        count = nginx_collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    # 3. Count for GET method with path '/status'
    status_check_count = nginx_collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_check_count} status check")

if __name__ == "__main__":
    log_stats()
