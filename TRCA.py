#! /use/bin/env python
''' **************************************
# Author       :leo-Ne
# Last modified:2020-12-08 16:16
# Email        : leo@email.com
# Filename     :TRCA.py
# Description  : 
**************************************'''
from __future__ import print_function
import sys

import numpy as np
from numpy import linalg as LA
from scipy.io import loadmat, savemat
from scipy import signal as SIG
import matplotlib.pyplot as plt
import mne

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
#errInfo = r'Error: <filter para in SSVEPFilter() is not defined>'
#sys.exit(errInfo)

class TRCA():
    def __init__(self, _Subject=None, fs=250):
        self._eegData   = None
        dataDescription = {
                'shape':    None,
                'nChannel': None,
                'nSample':  None,
                'nEvent':   None,
                'nTrial':   None,
                'nBlock':   None
                }
        self._dataDescription= dataDescription.copy()
        # [nChennel, nSample, nEvent, nTrial, nBlock]
        self._trainData = None
        self._testData  = None
        self._W         = None
        self._result    = None
        self._label     = None      # Equal to event
        # TRCA Setting
        self._begin     = 0.14
        self._tuse      = 1.0
        self._fs        = fs
        del dataDescription
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
            shape = np.shape(dataUse)
            self._dataDescription['shape']    = shape
            self._dataDescription['nChannel'] = shape[0]
            self._dataDescription['nSample']  = shape[1]
            self._dataDescription['nEvent']   = shape[2]
            self._dataDescription['nTrial']   = shape[2]
            self._dataDescription['nBlock']   = shape[3]
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
    
    def SSVEPFilter(self, filterType=0):
        """
        type
          0: nomal trca
          1: enhance trca
        """
        data = self._eegData.copy()
        fs = self._fs
        dataFiltered = None
        if filterType == 0:
            Wn = [6.0, 90.0]
            Wn = np.array(Wn, np.float64) / (fs/2)
            b, a = SIG.cheby1(4, 0.1, Wn,btype="bandpass",analog=False,output='ba')
            dataFiltered = SIG.lfilter(b, a, data, axis=1)
            del b, a ,Wn
        elif filterType == 1:
            sys.exit("Error:<filterType=1 means use Enhance trca by adding filter bank, to which Leo was lazy!!!>")
        self._trainData = dataFiltered[:, :, :, :-1]
        self._testData = dataFiltered[:, :, :, -1][:, :, :, None]
        del data, dataFiltered
        return 

    def trca1(self):
        trainData      = self._trainData.copy()
        trainDataShape = np.shape(trainData)
        nChannels      = trainDataShape[0]
        nSamples       = trainDataShape[1]
        nEvents        = trainDataShape[2]
        nBlocks        = trainDataShape[3]
        nTrials        = nBlocks * 1    # 1 trails for each task in each block.
        # process
        W         = np.zeros([nEvents, nChannels], np.float64)
        Q         = np.zeros([nChannels, nChannels], np.float64)
        S         = np.zeros_like(Q)
        trainData = trainData - np.mean(trainData, axis = 1)[:, None, :, :]
        for nEvent in range(nEvents):
            data = trainData[:, :, nEvent, :]
            UX   = np.reshape(data,[nChannels, nSamples * nTrials], order='C')
            Q    = np.matmul(UX, UX.T) / nTrials
            S    = np.zeros_like(Q)
            for xi in range(nTrials):
                for xj in range(nTrials):
                    if xi != xj:
                        data_i  = data[:, :, xi]
                        data_j  = data[:, :, xj]
                        S      += np.matmul(data_i, data_j.T)
            S = S / (nTrials * (nTrials-1))  
            eigenvalues, eigenvectors = LA.eig(np.matmul(LA.inv(Q), S))
            w_index = np.max(np.where(eigenvalues == np.max(eigenvalues)))
            W[nEvent, :] = eigenvectors[:, w_index].T
            print("w_index: ", nEvent, '\t', w_index)
        self._W = W.copy()
        del trainData, nChannels, nSamples, nEvents, nTrials, trainDataShape, W, Q, S
        del UX, data, data_i, data_j, eigenvalues, eigenvectors, w_index 
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
        W = self._W.copy()
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
    session.loadData(filename=sub6)
    session.cutData()
    print("eegData shape:\t", np.shape(session._eegData))
    session.SSVEPFilter()
    session.trca1()
    pass

if __name__ == "__main__":
    unitTest()
    pass
