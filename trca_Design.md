# TRCA Class Illustration.  
## TRCA Class Aim:  
Design a learner to create a classifier by using TRCA and LDA.  

## Class method and function.  
* trca.loadData()
* trca.SSVEPFilter()
* trca.trca1()
* trca.trca2()
* trca.LDA()
* trca.train()
* trca.classifier()
* trca.output()
* TRCA.unitTest()  "In TRCA.py file, unitTest() is not a method of trca class, but a test function for trca class."

## Class data struct
* eegData
>> All SSVEP data for offline processing.
* trainData
* testData
* nEvent, nSample, nBlock, nTrial
>> Data size.
* W
>> Store space filters for all events.
* result
>> Store trca.test() output.


