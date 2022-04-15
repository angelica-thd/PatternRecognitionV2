import numpy as np
from tqdm import tqdm

class BettingCompanyNeuralNetwork:
    def __init__(self):
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = (np.array([]) for i in range(4))

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
        np.random.seed(1)
        # Randomly initializing weights
        self.w1 = 2 * np.random.random((10, 4)) - 1
        self.w2 = 2 * np.random.random((10, 10)) - 1
        self.w3 = 2 * np.random.random((3, 10)) - 1


    def fit(self, iterations=1000):
        # Iterating over training samples
        print("Training...")
        for i in tqdm(range(iterations)):               #loading bar
            for X, y in zip(self.Xtrain ,self.ytrain):
                # Feedforward
                a1, a2, a3, a4 = self.feedforward(X)
                
                # Backpropagation
                d2, d3, d4 = self.backpropagation(a2, a3, a4, y)

                self.w1 += np.dot(np.transpose(np.asmatrix(d2)), np.asmatrix(a1))
                self.w2 += np.dot(np.transpose(np.asmatrix(d3)), np.asmatrix(a2))
                self.w3 += np.dot(np.transpose(np.asmatrix(d4)), np.asmatrix(a3))
        print("\nTraining complete!\n")

    # Feedforward to find a2, a3, a4
    def feedforward(self, X):
        a1 = X
        z2 = np.dot(self.w1, a1)
        a2 = self.sigmoid(z2)
        z3 = np.dot(self.w2, a2)
        a3 = self.sigmoid(z3)
        z4 = np.dot(self.w3, a3)
        a4 = self.sigmoid(z4)
        return (a1, a2, a3, a4)

    # Backpropagation to find d2, d3, d4
    def backpropagation(self, a2, a3, a4, y):
        d4 = np.subtract(a4, y)
        d3 = np.multiply(np.dot(np.transpose(self.w3), d4), np.multiply(a3, 1-a3))
        d2 = np.multiply(np.dot(np.transpose(self.w2), d3), np.multiply(a2, 1-a2))
        return (d2, d3, d4)

    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.feedforward(sample)[3]
            predictions.append(prediction)
        return predictions

    def calculateAccuracy(self, predictions, y):
        accuracy = 0
        for prediction, output in zip(predictions, y):
            accuracy += list(np.subtract(prediction, output)).count(0) / len(prediction)
        accuracy /= len(predictions)
        return accuracy

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))