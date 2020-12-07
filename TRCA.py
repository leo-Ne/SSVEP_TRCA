#! /use/bin/env python
''' **************************************
# Author       :leo-Ne
# Last modified:2020-12-06 16:32
# Email        : leo@email.com
# Filename     :TRCA.py
# Description  : 
**************************************'''

import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal
import matplotlib.pyplot as plt
import mne

class TRCA():
    def __init__(self, _Subject=None, fs=250):
        self._eegData   = None
        self._dataShape = np.zeros([5],np.int32)
        # [nChennel, nSample, nEvent, nTrial, nBlock]
        self._trainData = None
        self._testData  = None
        self._W         = None
        self._result    = None
        self.label      = None      # Equal to event
        # TRCA Setting
        self._begin     = 0.14
        self._tuse      = 1.0
        self._fs        = fs
        pass

    def loadData(self, filename):
        # process
        data, dataUse = None, None 
        if self._eegData is None:
            data = loadmat(filename)['data']
            chnls = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62], np.int32)
            dataUse = data[chnls, :, :, :].copy()
            self._eegData = dataUse.copy()
            # filter
            print('Data was firstly loaded!')
        else:
            print('Data was already loaded!')
        del data, dataUse
        return

    def cutData(self, tBegin=-1, tuse=-1):
        fs = self._fs
        data = self._eegData.copy()
        if tBegin == -1 or tuse == -1 :
            tBegin = self._begin
            tEnd   = tBegin + self._tuse
            print('Data cut by default!')
        else:
            tEnd   = tBegin + tuse
            self._begin = tBegin
            self._tuse  = tuse
            print('Data cut by setting!')
        nBegin = np.int32(fs * tBegin)
        nEnd   = np.int32(fs * tEnd)
        dataUse = data[:, nBegin:nEnd, :, :]
        self._eegData = dataUse.copy()
        del data, nBegin, nEnd, dataUse
        return  
    
    def SSVEPFilter(self, b=None, a=None):
        data = self._eegData.copy()
        # process
        self.trainData = None
        self.testData = None
        del data
        return 

    def trca1(self):
        data = self.trainData.copy()
        #  process
        self.W = None
        del data
        return 

    def trca2(self):
        pass

    def train():
        # Step of train model may execute in the TRCA step.  
        # This function may be not used.
        pass

    def LDA(self):
        trainData = self.trainData.copy()
        testData  = self.testData.copy()
        W = self.W.copy()
        # process
        del trainData, testData, W
        return 

    def classifier(self):
        data = self._eegData.copy()
        trainData = None
        testData = None
        # process
        self.result = None
        del data, trainData, testData
        return 

    def output(self):
        result = self.result.copy()
        # output 
        del result
        return 

def unitTest():
    # Unit test
    sub6 = r'./tsing/S6.mat'
    session = TRCA(_Subject=6,fs=250)
    print("eegData shape:\t", np.shape(session._eegData))
    session.loadData(filename=sub6)
    print("eegData shape:\t", np.shape(session._eegData))
    print("tBegin tEnd fs:\t", session._begin, session._tuse, session._fs)
    session.cutData()
    print("eegData shape:\t", np.shape(session._eegData))
    pass

if __name__ == "__main__":
    unitTest()
    pass
