import numpy as np

class betCompany:
    def __init__(self, name):   
        self.name = name
        self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.w = (np.array([]) for i in range(5))

    def prepareData(self, data):
        X = data[:,:3]
        goals = data[:,3:]
        results = np.array([goalRow[0] - goalRow[1] for goalRow in goals])
        y = np.array([[1,0,0] if r > 0 else [0,1,0] if r == 0 else [0,0,1] for r in results])

        # Adding bias column X0
        X = np.column_stack((np.ones(X.shape[0]), X))

        return (X, y)

    def loadData(self, trainData, testData):
        self.Xtrain, self.ytrain = self.prepareData(trainData)
        self.Xtest, self.ytest = self.prepareData(testData)
        #Initializing weights
        self.resetWeights()


    def resetWeights(self):
        self.w = np.array([np.zeros(self.Xtrain.shape[1] - 1) for i in range(self.Xtrain.shape[1])])

    def predict(self,X):
        p = np.array(np.matmul(X, self.w))

        
        for i in range(p.shape[0]):
            row = p[i,:]
            outcome = list(row).index(max(row))
            if outcome == 0:
                p[i,:] = [1,0,0]
            elif outcome == 1:
                p[i,:] = [0,1,0]
            else:
                p[i,:] = [0,0,1]
        
        return p

    def calculateAccuracy(self, predictions, y):
        correct = 0
        for i in range(predictions.shape[0]):
            correct += (predictions[i,:] == y[i,:]).all()

        return correct / predictions.shape[0]