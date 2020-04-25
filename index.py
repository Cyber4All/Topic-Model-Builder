import numpy as np

from machine_learning.text_cleaner import get_cleaned_description_corpus, create_X_and_Y_variable_lists, load_word_vectors, get_tokenized_sequences, pad_tokenized_sequences, fill_pre_trained_embeddings
from machine_learning.keras_model_builder import build_cnn_text_classifier, build_rnn_text_classifier, build_bidirectional_rnn_classifier

from external_service_drivers.mongodb_driver import get_unique_topic_list
from external_service_drivers.elasticsearch_driver import get_learning_objects_in_topic

from dotenv import load_dotenv

load_dotenv()

def start():
    word2vec = load_word_vectors()

    unique_topics_list = get_unique_topic_list()

    descriptions, topic_vectors = create_X_and_Y_variable_lists(unique_topics_list)

    cleaned_descriptions = get_cleaned_description_corpus(descriptions)

    tokenized_sequences, word2idx = get_tokenized_sequences(cleaned_descriptions)

    data = pad_tokenized_sequences(tokenized_sequences)

    num_words, embedding_matrix = fill_pre_trained_embeddings(word2idx, word2vec)

    possible_labels = np.asarray(unique_topics_list)

    build_cnn_text_classifier(num_words, embedding_matrix, possible_labels, data, topic_vectors)

    build_rnn_text_classifier(num_words, embedding_matrix, possible_labels, data, topic_vectors)

    build_bidirectional_rnn_classifier(num_words, embedding_matrix, possible_labels, data, topic_vectors)

if __name__ == '__main__':
    start()