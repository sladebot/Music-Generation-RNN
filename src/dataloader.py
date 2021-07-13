import numpy as np
import regex as re
import os
from pydub import AudioSegment
from pydub.playback import play

cwd = os.path.dirname(__file__)


class DataLoader:
    def __init__(self, seq_len, batch_size, dataset_path):
        self.dataset_path = dataset_path
        self.songs = self.load_training_data()
        self.songs_joined = "\n\n".join(self.songs)
        self.vocab = sorted(set(self.songs_joined))
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        self.seq_len = seq_len
        self.batch_size = batch_size

    def get_idx(self):
        return self.char2idx, self.idx2char

    def get_vocab(self):
        return self.vocab

    def play_song_idx(self, index):
        self.play_song(self.songs[index])

    def load_training_data(self):
        with open(self.dataset_path, "r") as f:
            text = f.read()
        songs = self.extract_song_snippet(text)
        return songs

    def extract_song_snippet(self, text):
        pattern = '(^|\n\n)(.*?)\n\n'
        search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
        songs = [song[1] for song in search_results]
        print("Found {} songs in text".format(len(songs)))
        return songs

    def vectorize_string(self, string):
        vectorized = np.array([self.char2idx[s] for s in string])
        return vectorized

    def get_batch(self):
        vectorized_songs = self.vectorize_string(self.songs_joined)
        n = vectorized_songs.shape[0] - 1
        idx = np.random.choice(n - self.seq_len, self.batch_size)
        input_batch = [vectorized_songs[i: i + self.seq_len] for i in idx]
        output_batch = [vectorized_songs[i + 1: i + self.seq_len + 1] for i in idx]
        x_batch = np.reshape(input_batch, [self.batch_size, self.seq_len])
        y_batch = np.reshape(output_batch, [self.batch_size, self.seq_len])
        return x_batch, y_batch

    def extract_song_snippet(self, text):
        pattern = '(^|\n\n)(.*?)\n\n'
        search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
        songs = [song[1] for song in search_results]
        print("Found {} songs in text".format(len(songs)))
        return songs

    def play_song(self, song):
        basename = self.save_song_to_abc(song)
        ret = self.abc2wav(basename + '.abc')
        if ret == 0:
            return self.play_wav(basename + '.wav')
        return None

    def play_wav(self, wav_file):
        song = AudioSegment.from_wav(wav_file)
        play(song)

    def abc2wav(self, abc_file):
        path_to_tool = os.path.join(cwd, 'bin', 'abc2wav')
        cmd = "{} {}".format(path_to_tool, abc_file)
        return os.system(cmd)

    def save_song_to_abc(self, song, filename="tmp"):
        save_name = "{}.abc".format(filename)
        with open(save_name, "w") as f:
            f.write(song)
        return filename




