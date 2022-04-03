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

    def fit(self, k=10, type='nonlinear',iterations=1000):
        # Iterating over training samples
        print("Training...")
        for i in tqdm(range(iterations)):               #loading bar
            for X, y in zip(self.Xtrain ,self.ytrain):
                # Feedforward
                a1, a2, a3, a4 = self.feedforward(X,type)
            
                # Backpropagation
                d2, d3, d4 = self.backpropagation(a2, a3, a4, y)

                #TODO:to be assigned
                #self.w1 += np.transpose(np.dot(np.transpose(np.asmatrix(a1)), np.asmatrix(d2)))
                #self.w2 += np.transpose(np.dot(np.transpose(np.asmatrix(a2)), np.asmatrix(d3)))
                #self.w3 += np.transpose(np.dot(np.transpose(np.asmatrix(a3)), np.asmatrix(d4)))

                if k%2==0:
                    self.w1 += np.nan_to_num(np.transpose(np.dot(np.transpose(np.asmatrix(a1)), np.asmatrix(d2))))
                    self.w2 += np.nan_to_num(np.transpose(np.dot(np.transpose(np.asmatrix(a2)), np.asmatrix(d3))))
                    self.w3 += np.nan_to_num(np.transpose(np.dot(np.transpose(np.asmatrix(a3)), np.asmatrix(d4))))
                else: 
                    self.w1 += np.transpose(np.dot(np.transpose(np.asmatrix(a1)), np.asmatrix(d2)))
                    self.w2 += np.transpose(np.dot(np.transpose(np.asmatrix(a2)), np.asmatrix(d3)))
                    self.w3 += np.transpose(np.dot(np.transpose(np.asmatrix(a3)), np.asmatrix(d4)))
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
            a2 = np.dot(self.w1, a1)
            a3 = np.dot(self.w2, a2)
            a4 = np.dot(self.w3, a3)

        (a1,a2,a3,a4) = (np.nan_to_num(a1), np.nan_to_num(a2), np.nan_to_num(a3), np.nan_to_num(a4))
        return (a1, a2, a3, a4)

    # Backpropagation to find d2, d3, d4
    def backpropagation(self, a2, a3, a4, y):
        d4 = np.subtract(a4, y)
        i3 = np.ones(a3.shape)
        i2 = np.ones(a2.shape)

        #TODO: to be assigned
        #gz3 = np.dot(a3, i3-a3)
        #gz2 = np.dot(a2, i2-a2)
        #theta3 = np.nan_to_num(np.dot(np.transpose(self.w3), d4))
        #d3 = np.nan_to_num(np.dot(theta3, gz3)) 
        #theta2 = np.nan_to_num(np.dot(np.transpose(self.w2), d3))
        #d2 = np.nan_to_num(np.dot(theta2, gz2))     

        gz3 = np.nan_to_num(np.dot(a3, i3-a3))
        gz2 = np.nan_to_num(np.dot(a2, i2-a2))

        theta3 = np.nan_to_num(np.dot(np.transpose(self.w3), d4))
        d3 = np.nan_to_num(np.dot(theta3, gz3)) 
        
        theta2 = np.nan_to_num(np.dot(np.transpose(self.w2), d3))
        d2 = np.nan_to_num(np.dot(theta2, gz2))
        
        return (d2, d3, d4)

    def predict(self, X, type='nonlinear'):
        predictions = []
        for sample in X:
            prediction = self.feedforward(sample,type)[3]
            predictions.append(prediction)
        return predictions

    def calculateAccuracy(self, predictions, y):
        accuracy = 0
        for prediction, output in zip(predictions, y):
            l = np.subtract(prediction, output)
            accuracy += np.count_nonzero(l == 0) / len(prediction)
        accuracy /= len(predictions)
        return accuracy

    # Sigmoid activation function: NON LINEAR
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

 