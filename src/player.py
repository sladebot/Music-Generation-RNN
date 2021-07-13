import regex as re

from pydub import AudioSegment
from pydub.playback import play
import os


def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs


def save_song_to_abc(song, filename="tmp"):
    save_name = "{}.abc".format(filename)
    with open(save_name, "w") as f:
        f.write(song)
    return filename


def play_song(song):
    basename = save_song_to_abc(song)
    ret = abc2wav(basename + '.abc')
    if ret == 0:
        return play_wav(basename + '.wav')
    return None


def play_wav(wav_file):
    song = AudioSegment.from_wav(wav_file)
    play(song)


def abc2wav(abc_file):
    cwd = os.path.dirname(__file__)
    path_to_tool = os.path.join(cwd, 'bin', 'abc2wav')
    cmd = "{} {}".format(path_to_tool, abc_file)
    return os.system(cmd)