#!/usr/bin/env python3
"""
A function that inserts a new document
in a collection based on kwargs
"""


def insert_school(mongo_collection, **kwargs):
    """ Insert document in a collection based on kwargs

    Args:
        mongo_collection (_type_): _description_
    Returns the new _id
    """
    new_id = mongo_collection.insert_one(kwargs).inserted_id
    return new_id   