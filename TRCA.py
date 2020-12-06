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

class trca():
    def __init__(self, _Subject=None):
        self.eegData   = None
        self.dataShape = np.zeros([5],np.int32)
        # [nChennel, nSample, nEvent, nTrial, nBlock]
        self.trainData = None
        self.testData  = None
        self.W         = None
        self.result    = None
        pass

    def loadData(self, filename):
        data = loadmat(filename)
        # process
        self.eegData = data.copy()
        del data
        return
    
    def SSVEPFilter(self):
        data = self.eegData.copy()
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

    def classifier(self)
        data = self.eegData.copy()
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
    
    pass

if if __name__ == "__main__":
    unitTest()
    return
