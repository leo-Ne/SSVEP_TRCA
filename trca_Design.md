# TRCA Class Illustration.  
## TRCA Class Aim:  
Design a learner to create a classifier by using TRCA and LDA.  

## Class method and function.  
* trca.loadData()
* trca.cutData()
* trca.SSVEPFilter()
* trca.trca1()
* trca.trca2()
* trca.LDA()
* trca.train()
* trca.classifier()
* trca.output()
* TRCA.unitTest()  "In TRCA.py file, unitTest() is not a method of trca class, but a test function for trca class."

## Class data struct
* _eegData
>> All SSVEP data for offline processing.
* _trainData
* _testData
* _label
* _dataDescription
>> Shape
>> nEvent, nSample, nBlock, nTrial
* W
>> Store space filters for all events.
* _result
>> Store trca.test() output.
* _begin
* _fs   "When it inits the TRCA class, it will init the class with fs value."


