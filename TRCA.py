# author: leo
# date: 2020/11/11
# python version: 3.8
# Code version: 1.0
"""
SSVEP EEG form Tsing database: www.bci.med.tsing.edu.cn/download.html
EEG Data: [Electrode index, Time points, Target indx, Block index]
"""

import numpy as np
from scipy.io import loadmat, savemat
import mne


class TRCA:
    def __init__(self):
        """
        Initialize the class TRCA and load eeg data;
        Load train data only with one block;
        """
        # Initilization with None and 0
        self.eegData, self.freq_phase = None, None
        chans, smpls, trials, block   = 0, 0, 0, 0
        self.shape                    = np.array([chans, smpls, trials, block], np.int32)
        # some other para 
        self.W      = np.zeros([smpls], np.float32)
        self.models = None
        print('Initialization Done!')
        pass

    def loadData(self, eegFile:str, Freq_Phase:str, block=0):
        """
        Create a dictory variable by loading *.mat file which include eeg data and labels
        """
        eegData = loadmat(eegFile)
        eegData = np.array(eegData['data'], np.float32)

        freq_phase = loadmat(Freq_Phase)
        freq_phase = np.array([freq_phase['freqs'], freq_phase['phases']], np.float32)
        freq_phase = freq_phase[:, 0, :].T
        print(r'freq_phase shape:', '\t', np.shape(freq_phase))
        print(r'SSVEP EEG shape:', '\t', np.shape(eegData))
        self.eegData, self.freq_phase = eegData, freq_phase
        self.shape                    = eegData.shape
        return eegData, freq_phase

    def trca(self):
        """
        Firstly, calculate the CovMax S.
        Secondly, calculate the coefficients W by Rayleigh-Ritz method.
        """
        eegData = self.eegData.copy()
        chans, smpls, trials = self.shape
        print("shape:\t", chans, smpls, trials)
        S = np.zeros([chans])
        for trial_i in range(trials):
            x1 = eegData[:, :, trial_i]
            x1 = x1 - np.mean(x1, axis=1)
            for trial_j in range(trial_i+1, trials):
                x2 = eegData[:, :, trial_j]
                x2 = x2 - np.mean(x2, axis=1)
                S  += np.matmul(x1, x2.T) + np.matmul(x2, x1.T)
        ## end for
        ## Those code is working in a ineffective way! change it!
        UX = np.reshape(eegData,[chans, smpls*trials])
        UX = UX - np.mean(UX, axis=1)
        Q = np.matmul(UX, UX.T)
        QS = np.matmul(S, Q)
        # Q-1 * S maybe lead the wrong result!
        eigenvalues, eigenvectaors = np.linalg.eig(QS)
        maxEigenvealue_i = np.argwhere(eigenvalues==np.max(eigenvalues))
        w = eigenvectaors[maxEigenvealue_i]
        self.W += w
        return w
       
    def model(self):
        """
        By using the training data and W, function create models for classification.
        This step to decrease the computational requirements when TRCA works in hard device.
        """
        self.models = models
        return models

    def classification(self, testData:str, model=None):
        if model is None:
            self.model()
        pass


def testUnit():
    dirName       = r'./tsing/'
    eegFile       = r'S6.mat'
    labelFile     = r'Freq_Phase.mat'
    dataFileName  = dirName + eegFile
    labelFileName = dirName + labelFile
    session       = TRCA()
    session.loadData(dataFileName, labelFileName, block=0)
    pass


if __name__ == "__main__":
    testUnit()

