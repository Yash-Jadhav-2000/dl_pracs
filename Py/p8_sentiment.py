from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
import numpy as np

vocab_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

word_idx = imdb.get_word_index()
word_idx = {index: word for word, index in word_idx.items()}
print([word_idx[i] for i in x_train[0]])

print("Max length of review:: ", len(max((x_train+x_test), key=len)))
print("Min length of a review:: ", len(min((x_train+x_test), key=len)))

from tensorflow.keras.preprocessing import sequence

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

print(RNN_model.summary())

RNN_model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

history = RNN_model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_valid, y_valid))

print()
print("Simple RNN score: ", RNN_model.evaluate(x_test, y_test, verbose=1))