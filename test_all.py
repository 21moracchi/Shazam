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
    songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']

    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)

    def test_all(tmin, duration):

        for extract in database:  # pour chaque morceau on choisit un extrait
            encoder = Encoding(128, 120)
            filename = 'samples/' + extract['song']
            fs, s = read(filename)

            encoder.process(
                fs, s[int(tmin*fs):int(tmin*fs) + int(duration*fs)])
            hashes = encoder.signature
            candidats = []
            
            for song in database:  # on cherche le morceau que l'algorithme associe à cet extrait

                matching = Matching(song['signature'], hashes)

                if matching.does_it_match():
                    candidats.append(song['song'])
            # si c'est bien le bon morceau, l'algorithme continue
            if len(candidats) != 1 or candidats[0] != extract['song']:
                return f"l'extrait de {extract['song']} n'a pas été reconnu"
            
        # si tous les extraits ont été identifiés, on a réussi le test
        return "Le test a été réussi"

    print(test_all(40, 20))
