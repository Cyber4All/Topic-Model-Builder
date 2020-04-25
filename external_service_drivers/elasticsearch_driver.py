import json
import requests
import os

# Sends a query to Elasticsearch via REST API
# to get all documents with the matching
# "topic" field
def get_learning_objects_in_topic(topic):
    body = json.dumps({
        "size": 1000, 
        "query": {
            "bool": {
                "must": {
                    "term": {
                        "topic.keyword": topic
                    }
                }
            }
        }
    })

    es_response = requests.post(os.environ.get('ES_DOMAIN'), data=body, headers={'Content-Type': 'application/json'})
    
    learning_object_es_documents = json.loads(es_response.text)['hits']['hits']

    learning_objects = []

    for learning_object_es_document in learning_object_es_documents:
        
        learning_object = learning_object_es_document['_source']
        learning_objects.append(learning_object)


    return learning_objects