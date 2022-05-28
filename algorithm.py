"""
Algorithm implementation
"""
from errno import EEXIST
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------
# Create a fingerprint for an audio file based on a set of hashes
# ----------------------------------------------------------------------------


class Encoding:

    """
    Class implementing the procedure for creating a fingerprint 
    for the audio files

    The fingerprint is created through the following steps
    - compute the spectrogram of the audio signal
    - extract local maxima of the spectrogram
    - create hashes using these maxima

    """

    def __init__(self, nperseg, noverlap):
        """
        Class constructor

        To Do
        -----

        Initialize in the constructor all the parameters required for
        creating the signature of the audio files. These parameters include for
        instance:
        - the window selected for computing the spectrogram
        - the size of the temporal window 
        - the size of the overlap between subsequent windows
        - etc.

        All these parameters should be kept as attributes of the class.
        """
        self.nperseg = nperseg
        self.noverlap = noverlap

    def process(self, fs, s, min=1000):
        """

        To Do
        -----

        This function takes as input a sampled signal s and the sampling
        frequency fs and returns the fingerprint (the hashcodes) of the signal.
        The fingerprint is created through the following steps
        - spectrogram computation
        - local maxima extraction
        - hashes creation

        Implement all these operations in this function. Keep as attributes of
        the class the spectrogram, the range of frequencies, the anchors, the 
        list of hashes, etc.

        Each hash can conveniently be represented by a Python dictionary 
        containing the time associated to its anchor (key: "t") and a numpy 
        array with the difference in time between the anchor and the target, 
        the frequency of the anchor and the frequency of the target 
        (key: "hash")


        Parameters
        ----------

        fs: int
           sampling frequency [Hz]
        s: numpy array
           sampled signal
        """

        self.fs = fs
        self.s = s
        self.spectogram = spectrogram(
            self.s, self.fs, noverlap=self.noverlap, nperseg=self.nperseg)
        f, t, Sxx = self.spectogram
        #self.energy = np.sum(Sxx)
        #Sxx_sorted = np.sort(Sxx)
        #somme = 0
        # for elem in Sxx_sorted :
        #somme += elem
        # if somme > self.energy:
        # pass #à compléter

        coords = peak_local_max(Sxx, exclude_border=False, min_distance=min)
        self.coords = np.array(
            [np.array([t[coord[1]], f[coord[0]]]) for coord in coords])
        print(len(coords))

        signature = list()
        delta_t = 2  # 1 ou 2 secondes
        delta_f = 1000  # 1000 à 5000 Hz
        for ancre in self.coords:
            for cible in self.coords:
                if cible[0]-ancre[0] < delta_t and (cible[0] > ancre[0] and
                                                    abs(cible[1]-ancre[1]) < delta_f):
                    signature.append(
                        {'t': ancre[0], 'hash': [cible[0]-ancre[0], ancre[1], cible[1]]})
        self.signature = signature

        # Insert code here

    def display_spectrogram(self):
        """
        Display the spectrogram of the audio signal
        """

        f, t, Sxx = self.spectogram

        plt.pcolormesh(t, f, np.log(Sxx), shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.scatter(self.coords[:, 0], self.coords[:, 1])
        plt.show()


# ----------------------------------------------------------------------------
# Compares two set of hashes in order to determine if two audio files match
# ----------------------------------------------------------------------------

class Matching:

    """
    Compare the hashes from two audio files to determine if these
    files match

    Attributes
    ----------

    hashes1: list of dictionaries
       hashes extracted as fingerprints for the first audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    hashes2: list of dictionaries
       hashes extracted as fingerprint for the second audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    matching: numpy array
       absolute times of the hashes that match together

    offset: numpy array
       time offsets between the matches
    """

    def __init__(self, hashes1, hashes2):
        """
        Class constructor

        Compare the hashes from two audio files to determine if these
        files match

        To Do
        -----

        Implement a code establishing correspondences between the hashes of
        both files. Once the correspondences computed, construct the 
        histogram of the offsets between hashes. Finally, search for a criterion
        based on the histogram that allows to determine if both audio files 
        match

        Parameters
        ----------

        hashes1: list of dictionaries
           hashes extracted as fingerprint for the first audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target

        hashes2: list of dictionaries
           hashes extracted as fingerprint for the second audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target
        """

        self.hashes1 = hashes1
        self.hashes2 = hashes2
        time_1 = []
        time_2 = []
        for dict_1 in self.hashes1:
            for dict_2 in self.hashes2:
                if dict_1['hash'] == dict_2['hash']:

                    time_1.append(dict_1['t'])
                    time_2.append(dict_2['t'])
        self.time_1 = time_1
        self.time_2 = time_2
        np_1 = np.array(self.time_1)
        np_2 = np.array(self.time_2)
        self.difference = np_1 - np_2

    def display_scatterplot(self):
        """
        Display through a scatterplot the times associated to the hashes
        that match
        """

        plt.scatter(self.time_1, self.time_2)
        plt.show()

    def display_histogram(self):
        """
        Display the offset histogram
        """
        plt.hist(self.difference, bins = 100 )
        plt.show()
    def does_it_match(self):
        ys, xs, z = plt.hist(self.difference, bins = 100)
        first_peak = 0
        second_peak = 0
        for peak in ys:
            if peak > first_peak:
                second_peak = first_peak
                first_peak = peak

            elif peak > second_peak:
                second_peak = peak
        if first_peak > 3 * second_peak and len(self.time_1) > 10 :
            return True
        else:
            return False


# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    encoder = Encoding(64, 32)
    encoder_2 = Encoding(64, 32)

    #fs, s = read('samples/Dark Alley Deals - Aaron Kenny.wav')
    #fs, s = read('samples/Jal - Edge of Water - Aakash Gandhi.wav')
    fs, s = read('samples/Cash Machine - Anno Domini Beats.wav')
    encoder.process(fs, s)

    #fs, s = read('samples/Dark Alley Deals - Aaron Kenny.wav')
    #fs, s = read('samples/Jal - Edge of Water - Aakash Gandhi.wav')
    encoder_2.process(fs, s[50*fs: 60*fs])
    # encoder_2.display_spectrogram()

    matching = Matching(encoder.signature, encoder_2.signature)
    matching.display_scatterplot()
    plt.show()
    matching.display_histogram()
    plt.show()
    print(matching.does_it_match())
    # encoder.display_spectrogram(display_anchors=True)
