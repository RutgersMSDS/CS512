import numpy as np
import time


# for now I random create training data, testdata and traininglabel
# I expect traininglabel as an array length 60,000
# I expect trainingdata as an matrix 1,680,000 (60000*28)*28
# I expect testdata as an matrix 280,000(10000*28)*28
class KnnClassifier:
    def __init__(self, trainingsize, traingingdata, traininglabel, testdata, k):
        self.trainingdata = traingingdata[:int(len(traingingdata) * trainingsize)]
        self.traininglabel = traininglabel[:int(len(traininglabel) * trainingsize)]
        self.testdata = testdata
        self.k = k

	#
    def subsampling(self):
        print(len(self.trainingdata), "traingingdata")
        self.trainingdata = self.trainingdata.reshape(-1, len(self.trainingdata), 28)[:, ::2, ::2]
        self.testdata = self.testdata.reshape(-1, len(self.testdata), 28)[:, ::2, ::2]
        self.testdata = self.testdata[0]
        self.trainingdata = self.trainingdata[0]
        print("Done for Subsampling, now there are: ")
        print(len(self.trainingdata), "traingingdata")
        print("each training data has", np.size(self.trainingdata, 1), "feature")

    def test(self, sb):
        tstime = 0
        tctime = 0
        if sb != 0:
            self.subsampling()
        self.testdata = self.testdata.reshape(int(len(self.testdata) / len(self.testdata[0])),
                                              len(self.testdata[0]) * len(self.testdata[0]))
        self.trainingdata = self.trainingdata.reshape(int(len(self.trainingdata) / len(self.trainingdata[0])),
                                                      len(self.trainingdata[0]) * len(self.trainingdata[0]))
        test_respone = {}
        print("testdatalen:", len(self.testdata))

        for i in range(len(self.testdata)):
            time1 = time.time()
            thisp = np.tile(self.testdata[i], (len(self.trainingdata), 1))
            thisp = ((thisp - self.trainingdata) ** 2) ** 0.5
            thisp = thisp.T
            thisp = sum(thisp)
            time2 = time.time()
            rank = np.argsort(thisp)
            firstten = []

			# picking top k values
            for j in range(self.k):  # k
                firstten.append(self.traininglabel[rank[j]])


            test_respone["value"] = np.bincount(np.array(firstten).astype(np.int64)).argmax()

            time3 = time.time()
            tstime = tstime + (time3 - time2)
            tctime = tctime + (time2 - time1)

            test_respone["sorting_time"] = tstime
            test_respone["calculation_time"] = tctime

        # for each predict, sort spend 0.004 sec
        # for each predict, calculation spend 0.55 sec
        print("sort time:", tctime)
        print("calculate time", tstime)

        return test_respone

# 	trainingdata=np.random.randint(0,2,(1680000,28))
# 	testdata=np.random.randint(0,2,(280000,28))
# 	########################subsampling feature from 28*28 to 14*14#######################
# 	sstrainingdata=trainingdata.reshape(-1,1680000,28)[:,::2,::2].reshape(60000,14*14)
# 	sstestdata=testdata.reshape(-1,280000,28)[:,::2,::2].reshape(10000,14*14)
# 	######################################################################################
# 	trainingdata=trainingdata.reshape(60000,28*28)
# 		testdata=testdata.reshape(10000,28*28)
# 		traininglabel=np.random.randint(0,10,60000)
# 		testres=[]
# 		for i in range(len(sstestdata)):
# 			time1=time.time()
# 			thisp=np.tile(sstestdata[i],(len(sstrainingdata),1))
# 			thisp=((thisp-sstrainingdata)**2)**0.5
# 			thisp=thisp.T
# 			thisp=sum(thisp)
# 			time2=time.time()
# 			rank=np.argsort(thisp)
# 			firstten=[]
# 			for j in range(10):#k
# 				firstten.append(traininglabel[rank[j]])
# 			testres.append(np.bincount(np.array(firstten)).argmax())
# 			time3=time.time()
# 		tstime=tstime+(time3-time2)
# 		tctime=tctime+(time2-time1)
# 		#for each predict, sort spand 0.004 sec
# 		#for each predict, calculation spend 0.55 sec
# 		print("sort time:", tctime)
# 		print("calculate time", tstime)
# #I expect predict 10,000 pictures less than 2 hours, which is acceptable.
