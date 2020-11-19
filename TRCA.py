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
import matplotlib.pyplot as plt
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
        # para about Models
        self.W = np.zeros([trials, chans])
        self.models = np.zeros([trials, smpls])
        pass

    def loadData(self, eegFile:str, labelFile:str, testBlock=-1):
        """
        Create a dictory variable by loading *.mat file which include eeg data and labels
        """
        eegData = loadmat(eegFile)
        eegData = np.array(eegData['data'], np.float32)
        freq_phase = loadmat(labelFile)
        freq_phase = np.array([freq_phase['freqs'], freq_phase['phases']], np.float32)
        freq_phase = freq_phase[:, 0, :].T
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
                xj  = xj - np.mean(xj, axis = 1)[:, None]
                S  += np.matmul(xi, xj.T) + np.matmul(xi, xj.T)
        ## end for
        ## Those code is working in a ineffective way! change it!

        UX    = np.reshape(trainData,[chans, smpls*trials])
        UX    = UX - np.mean(UX, axis = 1)[:, None]
        Q     = np.matmul(UX, UX.T)
        Q_inv = np.linalg.inv(Q)
        QS    = np.matmul(Q_inv, S)
        # cal w 
        eigenvalues, eigenvectors = np.linalg.eig(QS)
        maxEigenvealue_i          = np.argwhere(eigenvalues == np.max(eigenvalues))
        w      = eigenvectors[maxEigenvealue_i][0, 0]
        w      = w[:, None]
        return w
       
    def train(self, objFile:str, labelFile:str):
        """
        By using the training data and W, function create models for classification.
        This step to decrease the computational requirements when TRCA works in hard device.
        """
        print("----------------- In Function train() -----------------")
        ######################load data#################################
        if self.eegData is None:
            self.loadData(eegFile=objFile, labelFile=labelFile)
            print("Loading eeg Data...")
            eegData  = self.eegData.copy()
        else:
            eegData = self.eegData.copy()
        ######################cal WX####################################
        chans, smpls, _, _ = self.shape
        labels_length      = np.shape(self.freq_phase)[0]
        print("chans:\t", chans, "smpls:\t", smpls)
        print('label_length:\t', labels_length)
        W = np.zeros([labels_length, chans], np.complex64)
        for label in range(labels_length-38):
            w           = self.trca(label = label)
            W[label, :] = np.squeeze(w)[:]
            print("\rIn %d/40 template calculating" % (label+1), flush=True, end='')
        print('\nW shape:\t', W.shape)
        # models
        # cal average value of each signal in each trial and chanel of all blocks
        mean_val  = np.mean(eegData, axis=1)[:, None, :, :]
        eegData   = eegData - mean_val
        sumSignal = np.sum(eegData, axis=3)
        print('eegData shape:\t', eegData.shape)
        print('sumSignal shape:\t', sumSignal.shape)
        WX = np.zeros([labels_length, smpls], np.complex64)
        for i in range(labels_length):
            w     = W[i][None, :]
            x     = sumSignal[:, :, i]
            wx    = np.matmul(w, x)
            WX[i] = wx[:]
        self.models = WX

        return WX

    def classification(self, testData:str, model=None):
        if model is None:
            self.train()
        pass


def testUnit():
    dirName       = r'./tsing/'
    eegFile       = r'S6.mat'
    labelFile     = r'Freq_Phase.mat'
    dataFileName  = dirName + eegFile
    labelFileName = dirName + labelFile
    session       = TRCA()
    session.loadData(dataFileName, labelFileName)
#    session.trca(label=0)
    session.train(objFile=dataFileName, labelFile=labelFileName)
    print('\n\ntestUnit Passed!')
    pass


if __name__ == "__main__":
    testUnit()

