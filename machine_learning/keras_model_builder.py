from keras.layers import Dense, Input, GlobalMaxPooling1D, LSTM, GlobalMaxPool1D, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.optimizers import Adam
from keras.models import Model

from machine_learning.keras_model_trainer import train_model
from machine_learning.file_converter import save_model_to_file
from machine_learning.constants import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM

# Creates a CNN model with 
#  - 2 convolution layers
#  - 2 1D max pooling layers
#  - output Dense layer
# The model is then compiled, trained
# on the given topic data, and 
# saved as a file on disk
def build_cnn_text_classifier(num_words, embedding_matrix, possible_labels, data, topic_vectors):
    embedding_layer = build_embedding_layer(num_words, embedding_matrix)

    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(len(possible_labels), activation='sigmoid')(x)

    model = Model(input_, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    trained_model = train_model(model, data, topic_vectors)

    save_model_to_file(trained_model, './models/cnn_classifier.sav')


# Creates a RNN model with a single LSTM layer
# 
# The model is then compiled, trained
# on the given topic data, and 
# saved as a file on disk
def build_rnn_text_classifier(num_words, embedding_matrix, possible_labels, data, topic_vectors):
    embedding_layer = build_embedding_layer(num_words, embedding_matrix)

    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = LSTM(200, return_sequences=True)(x)
    x = GlobalMaxPool1D()(x)
    output = Dense(len(possible_labels), activation='sigmoid')(x)

    model = Model(input_, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.01),
        metrics=['accuracy']
    )

    trained_model = train_model(model, data, topic_vectors)

    save_model_to_file(trained_model, './models/rnn_classifier.sav')

# Creates a bidirectional RNN model with a single LSTM layer
# 
# The model is then compiled, trained
# on the given topic data, and 
# saved as a file on disk
def build_bidirectional_rnn_classifier(num_words, embedding_matrix, possible_labels, data, topic_vectors):
    embedding_layer = build_embedding_layer(num_words, embedding_matrix)

    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = Bidirectional( LSTM(200, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    output = Dense(len(possible_labels), activation='sigmoid')(x)

    model = Model(input_, output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.01),
        metrics=['accuracy']
    )

    trained_model = train_model(model, data, topic_vectors)

    save_model_to_file(trained_model, './models/bidirectional_rnn_classifier.sav')

def build_embedding_layer(num_words, embedding_matrix):
    embedding_layer = Embedding(
        num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    )

    return embedding_layer