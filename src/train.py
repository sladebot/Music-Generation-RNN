from src.model import RNN
from src.dataloader import DataLoader
import numpy as np
import os
import tensorflow as tf


num_training_iterations = 2000  # Increase this to train longer
batch_size = 32  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

embedding_dim = 256
rnn_units = 1024  # Experiment between 1 and 2048

checkpoint_dir = '../training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


opt = tf.keras.optimizers.Adam(learning_rate)


summary_writer = tf.summary.SummaryWriter()


def vectorize_string(string, char2idx):
    vectorized = np.array([char2idx[s] for s in string])
    return vectorized


def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss


def train():
    dataset_path = os.path.join("../data", "irish.abc")
    dataloader = DataLoader(seq_len=100, batch_size=32, dataset_path=dataset_path)
    vocab = dataloader.get_vocab()
    rnn = RNN(vocab_size=len(vocab), embedding_dim=256, rnn_units=1024, batch_size=32)

    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    model = rnn.create()
    model.summary()
    history = []
    for iter in range(num_training_iterations):
        x_batch, y_batch = dataloader.get_batch()
        loss = train_step(x_batch, y_batch, model, opt).numpy().mean()
        history.append(loss)

        if iter % 100 == 0:
            print(f"Iteration: {iter}, Loss: {loss}")
            model.save_weights(checkpoint_prefix)
            tf.summary.scalar("training loss", loss)
    print("Finished training.")
    model.save_weights(checkpoint_prefix)


@tf.function
def train_step(x, y, model, optimizer):
    with tf.GradientTape() as tape:
        if model is None:
            raise Exception("No model defined")
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


if __name__ == "__main__":
    train()
