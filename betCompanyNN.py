import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


class NeuralNetworkBC:
    def __init__(self):   
        self.Xtrain, self.Xtest, self.ytrain, self.ytest, self.w = (np.array([]) for i in range(5))

    def prepareData(self, data):
        X = data[:,:3]
        goals = data[:,3:]
        results = np.array([goalRow[0] - goalRow[1] for goalRow in goals])
        y = np.array([[1,0,0] if r > 0 else [0,1,0] if r == 0 else [0,0,1] for r in results])

        # Adding bias column X0 -> b = [1,1,..,1]
        X = np.column_stack((np.ones(X.shape[0]), X)) # B = [X,b]
        self.X = X
        return (X, y)

    def loadData(self, trainData, testData):
        self.Xtrain, self.ytrain = self.prepareData(trainData)
        self.Xtest, self.ytest = self.prepareData(testData)
        #Initializing weights
        self.resetWeights()


    def resetWeights(self):
        np.random.seed(1)   
        self.w1 = 2 * np.random.random((10, 29)) - 1
        self.w2 = 2 * np.random.random((10, 10)) - 1
        self.w3 = 2 * np.random.random((3, 10)) - 1


    def fit(self, type='nonlinear', iterations=1000):
        # Iterating over training samples
        print("Training...")
        for i in tqdm(range(iterations)):               #loading bar
            for X, y in zip(self.Xtrain ,self.ytrain):
                if type == 'linear': 
                    # Feedforward
                    a2,a3,a4 = self.feedforward(X,'linear')
                    # Backpropagation
                    #dw = self.backpropagation(y, a=a,z=z, type = 'linear')
                    d2, d3, d4 = self.backpropagation(y, a2=a2, a3=a3, a4=a4)

                    #Update weights
                    #print(a.shape,da.shape,dw.shape,self.w.shape)
                    self.w1 += np.dot(np.transpose(np.asmatrix(d2)), np.asmatrix(a1))
                    self.w2 += np.dot(np.transpose(np.asmatrix(d3)), np.asmatrix(a2))
                    self.w3 += np.dot(np.transpose(np.asmatrix(d4)), np.asmatrix(a3))

                elif type == 'nonlinear': 
                    # Feedforward
                    a1, a2, a3, a4 = self.feedforward(X)
                    # Backpropagation
                    d2, d3, d4 = self.backpropagation(y, a2=a2, a3=a3, a4=a4)
                    self.w1 += np.dot(np.transpose(np.asmatrix(d2)), np.asmatrix(a1))
                    self.w2 += np.dot(np.transpose(np.asmatrix(d3)), np.asmatrix(a2))
                    self.w3 += np.dot(np.transpose(np.asmatrix(d4)), np.asmatrix(a3))
                else:  raise Exception('Non-supported Neural Network type')
               
        print("\nTraining complete!\n")
    
    # Feedforward to find a2, a3, a4
    def feedforward(self, X, type='nonlinear'):
        if type == 'linear':  #transfer f(x) = Wx+b
            print(X.shape,self.w1.shape)
            a2 = np.dot(self.w1, X)
            a3 = np.dot(self.w2, X)
            a4 = np.dot(self.w3, X)
            return (a2,a3,a4)

        elif type == 'nonlinear': #activation f(x) = sigmoid(x)
            a1 = X
            z2 = np.dot(self.w1, a1)
            a2 = self.sigmoid(z2)
            z3 = np.dot(self.w2, a2)
            a3 = self.sigmoid(z3)
            z4 = np.dot(self.w3, a3)
            a4 = self.sigmoid(z4)
            return (a1, a2, a3, a4)
        else:  raise Exception('Non-supported Neural Network type')
                    
      

    # Backpropagation to find d2, d3, d4
    def backpropagation(self, y, a = None, z=None, a2 = None, a3 = None, a4 = None):
        print(a4.shape,a3.shape,a2.shape,y.shape,self.w1.shape)
        d4 = np.subtract(a4, y)
        d3 = np.multiply(np.dot(np.transpose(self.w3), d4), np.multiply(a3, 1-a3))
        d2 = np.multiply(np.dot(np.transpose(self.w2), d3), np.multiply(a2, 1-a2))
        return (d2, d3, d4)
        
        

    def predict(self,X, t = 'nonlinear'):
        predictions = []
        for sample in X:
            print(self.feedforward(sample,t))

            prediction = self.feedforward(sample,t)[1]
            print(prediction)
            predictions.append(prediction)
        return predictions

    def calculateAccuracy(self, predictions, y):
        accuracy = 0
        for prediction, output in zip(predictions, y):
            l = np.subtract(prediction, output)
            print(prediction,output,l)
            print(len(prediction))
            print(np.count_nonzero(l == 0)/len(prediction))

            accuracy += np.count_nonzero(l == 0) / len(prediction)
        accuracy /= len(predictions)
        return accuracy
    
    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
  