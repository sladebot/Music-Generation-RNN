import tensorflow as tf

class RNN:
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size

    def create(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(
                self.vocab_size,
                self.embedding_dim,
                batch_input_shape=[self.batch_size, None]),

            tf.keras.layers.LSTM(
                self.rnn_units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                recurrent_activation='sigmoid',
                stateful=True),
            tf.keras.layers.Dense(self.vocab_size)
          ])