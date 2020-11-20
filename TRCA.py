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
    def __init__(self, _objFile:str, _labelFile:str):
        """
        Initialize the class TRCA and load eeg data;
        Load train data only with one block;
        """
        # file names
        self.objFile   = _objFile
        self.labelFile = _labelFile
        # Initilization with None and 0
        self.eegData    = None
        self.trainData  = None
        self.testData   = None
        self.freq_phase = None
        chans, smpls, trials, block = 0, 0, 0, 0
        self.shape                  = np.array([chans, smpls, trials, block], np.int32)
        # para about Models
        self.W      = None
        self.models = None
        pass

    def loadData(self, testBlock=-1):
        """
        Create a dictory variable by loading *.mat file which include eeg data and labels
        """
        print("Loading  files:\t")
        print(self.objFile)
        print(self.labelFile)
        print('testBlock:\t', testBlock)
        eegData    = self.eegData    
        freq_phase = self.freq_phase
        if self.eegData is None:
            eegData         = loadmat(self.objFile)
            eegData         = np.array(eegData['data'], np.float32)
        if self.freq_phase is None:
            freq_phase      = loadmat(self.labelFile)
            freq_phase      = np.array([freq_phase['freqs'], freq_phase['phases']], np.float32)
            freq_phase      = freq_phase[:, 0, :].T
        ######################slice data##############################
        trainData = None
        testData  = None
        _, _, _, blocks = eegData.shape
        if testBlock in range(0, blocks):
            testData = eegData[:, :, :, testBlock]
            if testBlock != 0 or testBlock != blocks:
                trainData = np.concatenate((eegData[:,:,:,:testBlock], eegData[:,:,:,testBlock+1:]), axis=3)
            elif testBlock == 0:
                trainData = eegData[:,:,:,1:]
            elif testBlock == blocks-1:
                trainData = eegData[:,:,:,blocks-1]
        else:
            trainData = eegData
        ######################slice data End##########################
        self.eegData    = eegData
        self.trainData  = trainData
        self.testData   = testData
        self.freq_phase = freq_phase
        self.shape      = eegData.shape
        return trainData, freq_phase

    def trca(self, label:int):
        """
        Firstly, calculate the CovMax S.
        Secondly, calculate the coefficients W by Rayleigh-Ritz method.
        For each label, W was stored as a vector(chans, 1).
        """
        # labels
        freq_phase = self.freq_phase.copy()
        trainData = self.trainData.copy()
        # SSVEP EEG for one freqs
        trainData  = trainData[:, :, label, :]
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
       
    def train(self):
        """
        By using the training data and W, function create models for classification.
        This step to decrease the computational requirements when TRCA works in hard device.
        """
        print("------------- In Function train() -------------")
        ######################load data#################################
        if self.trainData is None:
            self.loadData()
            print("Loading eeg Data...")
            trainData  = self.trainData.copy()
        else:
            print("Data already loaded!")
            trainData = self.trainData.copy()
        ######################cal WX####################################
        chans, smpls, _, _ = self.trainData.shape
        labels_length      = np.shape(self.freq_phase)[0]
#        print("chans:\t", chans, "smpls:\t", smpls)
#        print('label_length:\t', labels_length)
        W = np.zeros([labels_length, chans], np.complex64)
        for label in range(labels_length):
            w           = self.trca(label = label)
            W[label, :] = np.squeeze(w)[:]
            print("\rIn %d/40 template calculating" % (label+1), flush=True, end='')
        print('')
#        print('\nW shape:\t', W.shape)
        self.W = W
        # models
        # cal average value of each signal in each trial and chanel of all blocks
        mean_val  = np.mean(trainData, axis=1)[:, None, :, :]
        trainData   = trainData - mean_val
        sumSignal = np.sum(trainData, axis=3)
#        print('trainData shape:\t', trainData.shape)
#        print('sumSignal shape:\t', sumSignal.shape)
        WX = np.zeros([labels_length, smpls], np.complex64)
        for i in range(labels_length):
            w     = W[i][None, :]
            x     = sumSignal[:, :, i]
            wx    = np.matmul(w, x)
            WX[i] = wx[:]
        ######################cal WX END################################
        self.models = WX
        print("------------- End Function train() ------------")
        return WX

    def classification(self):
        """
        Classification() is to test generalization of TRCA model by using 5-1 floder.  
        """
#        if self.models is None:
#            print("There is not models! Please train it!")
#            return -1
#        _, _, _, blocks = self.shape
        blocks = 6
        testNum = 40
        print("->->->->test models<-<-<-<-")
        for testBlk in range(blocks):
            self.loadData(testBlock=testBlk)
            self.train()
            w = self.W
            WX = self.models
            testData = self.testData - np.mean(self.testData, axis=1)[:, None, :]
            outputLabels = np.zeros([testNum], np.int32)
            for test_i in range(testNum):
                data_t = testData[:, :, test_i]
                WT = np.matmul(w, data_t)
                coefficience = np.zeros([testNum])
                for i, wt in enumerate(WT):
                    wx = WX[i]
                    wt = np.abs(wt)
                    wx = np.abs(wx)
                    coefficienceMatrix = np.corrcoef(wt, wx)
                    coefficience[i] = coefficienceMatrix[0, 1]
                outputLabels[test_i] = np.argwhere(coefficience==np.max(coefficience))[:, :]
            print("outputLabels\t", outputLabels)
        print("->->->->test models end<-<-<-<-")

        return 1

def testUnit():
    print("----------------- In Function testUnit() -----------------")
    print("-----------------   Test Unit start      -----------------")
    dirName       = r'./tsing/'
    eegFile       = r'S6.mat'
    labelFile     = r'Freq_Phase.mat'
    dataFileName  = dirName + eegFile
    labelFileName = dirName + labelFile
    session       = TRCA(_objFile=dataFileName, _labelFile=labelFileName)
#    session.loadData(testBlock=1)
#    session.loadData(testBlock=2)
#    session.trca(label=0)
#    session.train()
    session.classification()
    print('testUnit Passed!')
    print("-----------------   Test Unit End        -----------------")
    pass


if __name__ == "__main__":
    testUnit()

