import re
import numpy as np
import os
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from machine_learning.constants import EMBEDDING_DIM, MAX_VOCAB_SIZE, MAX_SEQUENCE_LENGTH
from external_service_drivers.elasticsearch_driver import get_learning_objects_in_topic

# Removes markup, tokenizes, and stems
# the given data
def get_cleaned_description_corpus(descriptions):

    nltk.download('stopwords')

    corpus = []

    for i, description in enumerate(descriptions):
        cleanr = re.compile('<.*?>')
        cleaned_description = re.sub(cleanr, '', description)
        cleaned_description = re.sub('[^a-zA-Z]', ' ', description)
        cleaned_description = cleaned_description.lower()
        cleaned_description = cleaned_description.split()
        ps = PorterStemmer()
        cleaned_description = [ps.stem(word) for word in cleaned_description if not word in set(stopwords.words('english')) and len(word) > 1]
        cleaned_description = ' '.join(cleaned_description)
        corpus.append(cleaned_description)

    corpus = np.asarray(corpus)
    return corpus

# Creates the "topic vector". The topic vector
# is a 2D array where [n][n]
# where n = length of unique topic list
# The inner arrays consist of 0's and a 1
# at the index for a matching topic
# This function also stores all Learning Object descriptions
# in an array
def create_X_and_Y_variable_lists(unique_topics_list):
    descriptions = []
    topic_vectors = []

    for unique_topic in unique_topics_list:  

        topic_vector = []
        
        for unique_topic_for_matching in unique_topics_list:

            if unique_topic == unique_topic_for_matching:
                topic_vector.append(1)
            else:
                topic_vector.append(0)

        learning_objects = get_learning_objects_in_topic(unique_topic)

        for learning_object in learning_objects:

            descriptions.append(learning_object.get('description'))
            topic_vectors.append(topic_vector)

    topic_vectors = np.asarray(topic_vectors)
    return descriptions, topic_vectors
    

def load_word_vectors():
    word2vec = {}
    with open(os.path.join('./machine_learning/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec

    return word2vec

def get_tokenized_sequences(sentences):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)

    word2idx = tokenizer.word_index

    return sequences, word2idx

def pad_tokenized_sequences(sequences):
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return data

def fill_pre_trained_embeddings(word2idx, word2vec):
    num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx.items():
        if i < MAX_VOCAB_SIZE:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return num_words, embedding_matrix