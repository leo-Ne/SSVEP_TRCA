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
    def __init__(self, _Subject=None, begin=0.14, tuse=1.0, fs=250):
        self._eegData   = None
        self._dataShape = np.zeros([5],np.int32)
        # [nChennel, nSample, nEvent, nTrial, nBlock]
        self._trainData = None
        self._testData  = None
        self._W         = None
        self._result    = None
        self.label      = None      # Equal to event
        # TRCA Setting
        self._begin     = begin
        self._tuse      = tuse
        self.fs         = fs
        pass

    def loadData(self, filename):
        # process
        if self._eegData is None:
            data = loadmat(filename)['data']
            chnls = np.array([47, 53, 54, 55, 56, 57, 60, 61, 62], np.int32)
            dataUse = data[chnls, :, :, :].copy()
            # filter
            del data
        else:
            dataUse = self._eegData.copy()
        self._eegData = dataUse.copy()
        del dataUse
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
    session = TRCA(_Subject=6, begin=0.14, tuse=1,fs=250)
    session.loadData(filename=sub6)
    session.loadData(filename=sub6)
    session.loadData(filename=sub6)
    session.loadData(filename=sub6)
    session.loadData(filename=sub6)
    pass

if __name__ == "__main__":
    unitTest()
    pass
