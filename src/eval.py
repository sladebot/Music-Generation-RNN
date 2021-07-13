import tensorflow as tf
import os
from model import RNN
from src.dataloader import DataLoader
from player import extract_song_snippet, play_song

num_training_iterations = 2000  # Increase this to train longer
batch_size = 32  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

checkpoint_dir = '../training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


class Evaluator:
    def __init__(self):
        dataset_path = os.path.join("../data", "irish.abc")
        dataloader = DataLoader(seq_len=100, batch_size=1, dataset_path=dataset_path)
        vocab = dataloader.get_vocab()
        self.char2idx, self.idx2char = dataloader.get_idx()
        rnn = RNN(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024, batch_size=1)
        model = rnn.create()
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        model.build(tf.TensorShape([1, None]))
        print(model.summary())
        self.model = model
        self.dataloader = dataloader

    def generate_text(self, start_string, generate_length=1000):
        input_eval = [self.char2idx[ch] for ch in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        self.model.reset_states()

        for i in range(generate_length):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])

        return start_string + ''.join(text_generated)


if __name__ == "__main__":
    e = Evaluator()
    cwd = os.path.dirname(__file__)
    generated_text = e.generate_text("X", generate_length=1000)
    print(generated_text)
    generated_songs = extract_song_snippet(generated_text)
    for i, song in enumerate(generated_songs):
        waveform = play_song(song)







