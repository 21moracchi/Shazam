"""
Description
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from algorithm import *

# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    # 1: Load the database
    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)

    # 2: Create an instance of the class Encoder
    encoder = Encoding(128, 120)

    # 3: Randomly get an extract from one of the songs of the database
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
    song = random.choice(songs)
    print('Selected song: ' + song[:-4])
    filename = 'samples/' + song
    

    fs, s = read(filename)
    tmin = int(50*fs)  # We select an extract starting at 50s ...
    duration = int(10*fs)  # ... which lasts 10s

    # 4: Use the encoder to extract a fingerprint of the sample
    encoder.process(fs, s[tmin:tmin + duration])
    hashes = encoder.signature

    # 5: Using the class Matching, compare the fingerprint to all the
    # fingerprints in the database
    for song in database:
        print(song['song'])
        matching = Matching(song['signature'], hashes)

        matching.display_histogram()
        print(matching.does_it_match())
    # Insert code here
