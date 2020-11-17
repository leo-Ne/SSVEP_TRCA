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
        self.W      = np.zeros([chans], np.float32)
        self.models = None
        pass

    def loadData(self, eegFile:str, labelFile:str, block=0):
        """
        Create a dictory variable by loading *.mat file which include eeg data and labels
        """
        eegData = loadmat(eegFile)
        eegData = np.array(eegData['data'], np.float32)

        freq_phase = loadmat(labelFile)
        freq_phase = np.array([freq_phase['freqs'], freq_phase['phases']], np.float32)
        freq_phase = freq_phase[:, 0, :].T
        print(r'freq_phase shape:', '\t', np.shape(freq_phase))
        print(r'SSVEP EEG shape:', '\t', np.shape(eegData))
        print('labels:')
        print('\tfreqs: \tphases: \t')
        for i, item in enumerate(freq_phase):
            print('label'+str(i)+'\t', item[0], '\t', item[1])
        self.eegData, self.freq_phase = eegData, freq_phase
        self.shape                    = eegData.shape
        self.W                        = np.zeros([self.shape[2]], np.float32)
        return eegData, freq_phase

    def trca(self, label:int):
        """
        Firstly, calculate the CovMax S.
        Secondly, calculate the coefficients W by Rayleigh-Ritz method.
        For each label, W was stored as a vector(chans, 1).
        """
        # labels
        freq_phase = self.freq_phase.copy()
        eegData    = self.eegData.copy()
        # SSVEP EEG for one freqs
        trainData  = eegData[:, :, label, :]
        chans, smpls, trials = np.shape(trainData)
        S = np.zeros([chans, chans], np.float32)
        # calculate the matrix S
        for trial_i in range(trials):
            xi = trainData[:, :, trial_i]
            xi = xi - np.mean(xi, axis=1)[:, None]
            for trial_j in range(trial_i+1, trials):
                xj  = trainData[:, :, trial_j]
                xj  = x2 - np.mean(xj, axis = 1)[:, None]
                S  += np.matmul(xi, xj.T) + np.matmul(xi, xj.T)
        ## end for
        ## Those code is working in a ineffective way! change it!

        UX = np.reshape(trainData,[chans, smpls*trials])
        UX = UX - np.mean(UX, axis=1)[:, None]
        Q = np.matmul(UX, UX.T)
        Q_inv = np.linalg.inv(Q)
        QS = np.matmul(Q_inv, S)
        # cal w 
        eigenvalues, eigenvectors = np.linalg.eig(QS)
        maxEigenvealue_i = np.argwhere(eigenvalues==np.max(eigenvalues))
        w = eigenvectors[maxEigenvealue_i][0, 0]
        w = w[:, None]
        print("W : \t", w.T)
        print("W shape :\t", np.shape(w))
        self.W = w
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
    session.trca(label=0)
    print('testUnit Passed!')
    pass


if __name__ == "__main__":
    testUnit()

