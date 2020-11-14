# author: leo
# date: 2020/11/11
# python version: 3.8
# Code version: 1.0

import numpy as np
from scipy.io import loadmat, savemat
import mne


class TRCA:
    def __init__(self, _eegFile:str, Freq_Phase=None):
        """
        Initialize the class TRCA and load eeg data;
        Load train data only with one block;
        """
        self.eegData, self.freq_phase = self.loadData(_eegFile,Freq_Phase) 
        chans, smpls, trials = np.shape(self.eegData)
        self.shape = np.array([chans, smpls, trials], np.int32)
        # some other para 
        self.W = np.zeros([smpls], np.float32)
        self.models = None
        pass

    def loadData(self, eegFile:str, Freq_Phase=None):
        eegData = loadmat(eegFile)
        if Freq_Phase not None:
            freq_phase = loadmat(Freq_Phase)
        return eegData, freq_phase

    def trca():
        """
        Firstly, calculate the CovMax S.
        Secondly, calculate the coefficients W by Rayleigh-Ritz method.
        """
        eegData = self.eegData.copy()
        chans, smpls, trials = self.shape
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
#    session = TRCA(
    pass


if __name__ == "__main__":
    testUnit()

