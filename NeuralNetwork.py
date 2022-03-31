from time import sleep
import numpy as np
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self):
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = (np.array([]) for i in range(4))


    def prepareData(self, data):
        # Extracting X
        X = data[:,:-2]
        
        # Adding bias column x0
        X = np.column_stack((np.ones(X.shape[0]), X))
        
        # Extracting y
        goals = data[:,-2:]
        results = goals[:,0] - goals[:,1]
        y = np.array([[1,0,0] if result > 0 else [0,1,0] if result == 0 else [0,0,1] for result in results])

        return (X, y)

    def loadData(self, trainData, testData):
        self.Xtrain, self.ytrain = self.prepareData(trainData)
        self.Xtest, self.ytest = self.prepareData(testData)
        np.random.seed(1)
        # Randomly initializing weights
        self.w1 = 2 * np.random.random((10, 29)) - 1
        self.w2 = 2 * np.random.random((10, 10)) - 1
        self.w3 = 2 * np.random.random((3, 10)) - 1

    def fit(self, type='nonlinear', iterations=1000):
        # Iterating over training samples
        print("Training...")
        for i in tqdm(range(iterations)):               #loading bar
            for X, y in zip(self.Xtrain ,self.ytrain):
                # Feedforward
                a1, a2, a3, a4 = self.feedforward(X,type)
            
                # Backpropagation
                d2, d3, d4 = self.backpropagation(a2, a3, a4, y)

                self.w1 += np.dot(np.transpose(np.asmatrix(d2)), np.asmatrix(a1))
                self.w2 += np.dot(np.transpose(np.asmatrix(d3)), np.asmatrix(a2))
                self.w3 += np.dot(np.transpose(np.asmatrix(d4)), np.asmatrix(a3))
        print("\nTraining complete!\n")
    
    # Feedforward to find a2, a3, a4
    def feedforward(self, X, type='nonlinear'):
        #activation f(x) = sigmoid(x)
        if type == 'nonlinear':
            a1 = X
            z2 = np.dot(self.w1, a1)
            a2 = self.sigmoid(z2)
            z3 = np.dot(self.w2, a2)
            a3 = self.sigmoid(z3)
            z4 = np.dot(self.w3, a3)
            a4 = self.sigmoid(z4)
        elif type == 'linear':  #activation f(x) = Wx+b
            a1 = X
            a2 = np.dot(self.w1,a1)
            a3 = np.dot(self.w2,a2)
            a4 = np.dot(self.w3,a3)
            #print(a1,a2,a3,a4)
                    
        return (a1, a2, a3, a4)

    # Backpropagation to find d2, d3, d4
    def backpropagation(self, a2, a3, a4, y):
        d4 = np.subtract(a4, y)
        d3 = np.multiply(np.dot(np.transpose(self.w3), d4), np.multiply(a3, 1-a3))
        d2 = np.multiply(np.dot(np.transpose(self.w2), d3), np.multiply(a2, 1-a2))
        return (d2, d3, d4)

    def predict(self, X, type='nonlinear'):
        predictions = []
        for sample in X:
            prediction = self.feedforward(sample,type)[3]
            print(prediction)
            predictions.append(prediction)
        return predictions

    def calculateAccuracy(self, predictions, y):
        accuracy = 0
        for prediction, output in zip(predictions, y):
            l = np.subtract(prediction, output)
            print(l)
            print(np.count_nonzero(l == 0))
            accuracy += np.count_nonzero(l == 0) / len(prediction)
        accuracy /= len(predictions)
        return accuracy

    # Sigmoid activation function: NON LINEAR
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

 