#!/usr/bin/env python3
"""
List all documents
"""


def list_all(mongo_collection):
    """ Return a list of all documents

    Args:
        mongo_collection (mongocollection): Mongo collection

    Returns:
       empty list if no document in the collection
    """
    all_documents = mongo_collection.find()
    documents_list = list(all_documents)

    return documents_list
