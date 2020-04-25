from pymongo import MongoClient
import ssl
import os

client = MongoClient(os.environ.get('MONGO_DOMAIN'), ssl_cert_reqs=ssl.CERT_NONE)
learning_objects_collection = client.onion.objects
topics_collection = client.onion.topics

# Gets all of the topic names from the topics collection
# in MongoDB
# The list of topic names is stored within an array on
# a single document. Therefore, we can query with a
# find on operation
def get_unique_topic_list():
    # This collection should only eve have one document
    doc = topics_collection.find_one()

    unique_topics_list = doc.get('topics')

    return unique_topics_list