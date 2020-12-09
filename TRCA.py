#! /use/bin/env python
''' **************************************
# Author       :leo-Ne
# Last modified:2020-12-09 21:13
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
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import mne

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
#errInfo = r'Error: <filter para in SSVEPFilter() is not defined>'
#sys.exit(errInfo)

class TRCA():
    def __init__(self, _Subject=None, fs=250):
        self._Sub         = _Subject
        self._eegData     = None
        self._eegFiltered = None
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
        self._label     = None      # Equal to event
        self._result    = None
        # TRCA Setting
        self._begin     = 2.0
        self._tCut      = 0.14
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

    def cutData(self, tBegin=-1, tuse=-1, tCut=0.14):
        fs = self._fs
        data = self._eegData.copy()
        if tBegin == -1 or tuse == -1 :
            tBegin = self._begin + self._tCut
            tEnd   = tBegin + self._tuse
            print('Data cut by default!')
        else:
            self._begin = tBegin
            self._tuse  = tuse
            self._tCut  = tCut
            tBegin = tBegin + tCut
            tEnd   = tBegin + tuse
            print('Data cut by setting!')
        nBegin = np.int32(fs * (tBegin))
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
        data         = self._eegData.copy()
        fs           = self._fs
        dataFiltered = None
        if filterType == 0:
            Wn           = [7.0, 90.0]
            Wn           = np.array(Wn, np.float64) / (fs/2)
            b, a         = SIG.cheby1(4, 0.1, Wn,btype  = "bandpass",analog = False,output = 'ba')
#            dataFiltered = SIG.lfilter(b, a, data, axis = 1)
            dataFiltered = SIG.filtfilt(b, a, data, axis = 1)
            del b, a ,Wn
        elif filterType == 1:
            sys.exit("Error:<filterType=1 means use Enhance trca by adding filter bank, to which Leo was lazy!!!>")
        self._eegFiltered = dataFiltered.copy()
        del data, dataFiltered
        return
    
    def testSet(self, testBlock=0):
        nBlocks         = self._dataDescription['nBlock']
        if testBlock not in range(nBlocks):
            sys.exit('Error:<The test block setting is out range!!!>')
        trainBlock = list(range(nBlocks))
        trainBlock.remove(testBlock)
        trainBlock      = np.array(trainBlock, np.int32)
        eegData         = self._eegFiltered.copy()
        self._testData  = eegData[:, :, :, testBlock][:, :, :, None]
        self._trainData = eegData[:, :, :, trainBlock]
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
        self._W = W.copy()
        del trainData, nChannels, nSamples, nEvents, nTrials, trainDataShape, W, Q, S
        del UX, data, data_i, data_j, eigenvalues, eigenvectors, w_index 
        return 

    def trca2(self):
        pass
    

    def classifier(self):
        trainData   = self._trainData.copy()
        nEvents     = self._dataDescription['nEvent']
        testData    = self._testData.copy()
        nTestBlock  = np.shape(testData)[3]
        W           = self._W.copy()
        temp_X      = np.mean(trainData, axis = 3)

        coeffiience = np.zeros([nEvents], np.float32)
        result      = np.zeros([nEvents * nTestBlock], np.int32)
        for test_idx in  range(nEvents * nTestBlock):
            test_trial = testData[:, :, test_idx, 0]
            for i, w in enumerate(W):
                w = w[None, :]
                test_i = np.dot(w, test_trial)
                temp_i = np.dot(w, temp_X[:, :, i])
                coeffiience[i], _ = pearsonr(test_i[0], temp_i[0])
            
            label            = np.max(np.where(coeffiience == np.max(coeffiience)))
            result[test_idx] = label
        del trainData, nEvents, testData,nTestBlock, W, temp_X, coeffiience, test_trial, test_i, temp_i, label
        self._result = result.copy()
        del result
        return 

    def train(self,testBlock=0):
        self.testSet(testBlock=testBlock)
        self.trca1()
        return

    def output(self):
        result        = self._result.copy()
        correctResult = np.arange(0,40,1,dtype    = np.int32)
        tureNum       = np.size(np.where((result == correctResult) == True))
        accuracy      = tureNum / np.size(result)
        tBegin = self._begin + self._tCut
        tEnd  = tBegin + self._tuse
        del result, correctResult, tBegin, tEnd
        return tureNum, accuracy

def unitTest():
    # Unit test
    """
    Test code logic
    """
    sub        = r'./tsing/S1.mat'
    tBegin     = 3.0
    tCut       = 0.14
    tUse       = 1.0
    filterType = 0
    testBlock  = 1
    
    session = TRCA(_Subject=6,fs=250)
    session.loadData(filename=sub)
    session.cutData(tBegin,tUse,tCut)
    session.SSVEPFilter(filterType)

    session.train(testBlock)
    session.classifier()

    session.output()
    pass

def CVTest(SubNum=None):
    """
    Cross Validation for testing algorithmic ability.
    """
    if SubNum == None:
        sub = r'./tsing/S2.mat'
    else:
        sub = r'./tsing/S'+str(SubNum)+r'.mat'
    tBegin     = 1.0
    tCut       = 0.14
    tUse       = 1.0
    filterType = 0
    print('Subject:%s. Period of data:\t %.2f~%.2fs'% (sub, tBegin+tCut, tBegin+tCut+tUse))

    session = TRCA(_Subject=6,fs=250)
    session.loadData(filename=sub)
    session.cutData(tBegin,tUse,tCut)
    session.SSVEPFilter(filterType)
    averAcc = 0.0
    for loop in range(6):
        print('In loop: %d, test set: %d.' % (loop, loop), end=' ')
        session.train(testBlock=loop)
        session.classifier()
        tureNum, accuracy = session.output()
        print("tureNum:%d/%d, accuracy:%.2f%%" % (tureNum, 40, accuracy * 100))
        averAcc += accuracy
    print("Average accuracy:%.2f%%" % (100 * averAcc / 6))
    del sub
    return

def allCVTest():
    """
    Do CV by using all data of subjects.
    """
    SubNum = [1, 2, 4, 6, 7]
    print("================================================================")
    for subNum in SubNum:
        CVTest(SubNum=subNum)
        print("================================================================")

if __name__ == "__main__":
    allCVTest()
    pass
