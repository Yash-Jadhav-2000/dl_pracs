from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
import numpy as np

vocab_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

word_idx = imdb.get_word_index()
word_idx = {index: word for word, index in word_idx.items()}

max_words = 400
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)

x_valid, y_valid = x_train[:641], y_train[:641]
x_train, y_train = x_train[641:], y_train[641:]
embd_len = 32

RNN_model = Sequential(name="Simple_RNN")
RNN_model.add(Embedding(vocab_size, embd_len, input_length=max_words))
RNN_model.add(SimpleRNN(128, activation="tanh", return_sequences=False))
RNN_model.add(Dense(1, activation="sigmoid"))

RNN_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

history = RNN_model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_valid, y_valid))

print("Simple RNN score: ", RNN_model.evaluate(x_test, y_test, verbose=1))

gru_model = Sequential(name="GRU_Model")
gru_model.add(Embedding(vocab_size, embd_len, input_length=max_words))
gru_model.add(GRU(128, activation='tanh', return_sequences=False))
gru_model.add(Dense(1, activation='sigmoid'))

gru_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

history = gru_model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_valid, y_valid))

print("Simple GRU score: ", gru_model.evaluate(x_test, y_test, verbose=1))

lstm_model = Sequential(name="LSTM_Model")
lstm_model.add(Embedding(vocab_size, embd_len, input_length=max_words))
lstm_model.add(LSTM(128, activation='relu', return_sequences=False))
lstm_model.add(Dense(1, activation='sigmoid'))

lstm_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

history = lstm_model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_valid, y_valid))

print("Simple LSTM score: ", lstm_model.evaluate(x_test, y_test, verbose=1))