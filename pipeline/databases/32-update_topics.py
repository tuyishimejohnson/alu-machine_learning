#!/usr/bin/env python3
"""
 function that changes all topics of a
 school document based on the name
"""


def update_topics(mongo_collection, name, topics):
    """ Changes all topics of a school document based on the name

    Args:
        mongo_collection will be the pymongo collection object
        name (string) will be the school name to update
        topics (list of strings) will be the list of topics approached in the school
    """
    mongo_collection.update_many({"name": name}, {"$set": {"topics": topics}})