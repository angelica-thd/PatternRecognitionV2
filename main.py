from numpy import array, transpose, column_stack, ones
from sklearn.model_selection import KFold
from data import fetchData
from betCompany import betCompany
from LMS import LMSfit
from LS import LSfit
from NeuralNetwork import NeuralNetwork
from cMeans import cMeans

B365data, BWdata, IWdata, LBdata, TeamAttributesData = fetchData()
# Initializing betting companies
betCompanies = [betCompany("B365"), betCompany("BW"), betCompany("IW"), betCompany("LB")]

# Data dictionary
data = {"B365": B365data, "BW": BWdata, "IW": IWdata, "LB": LBdata}

# Initializing Kfold for 10fold
kf = KFold(n_splits=10)


'''
# Least Mean Squares Method
print('\n\n', 15 * '_' + 'LEAST_MEAN_SQUARES_METHOD' + 15 *'_', '\n')
for bc in betCompanies:
    iteration = 1
    averageAccuracy = 0
    for train_index, test_index in kf.split(data[bc.name]):        
        # Changing training and testing sets
        bc.loadData(data[bc.name][train_index], data[bc.name][test_index])
        
        # Calculating weights
        bc.w = LMSfit(bc.Xtrain, bc.ytrain)
        
        # Calculating accuracy
        accuracy = bc.calculateAccuracy(bc.predict(bc.Xtest) ,bc.ytest)
        averageAccuracy += accuracy / 10
        print(f'{bc.name} iteration {iteration} | Accuracy: {100 * accuracy}%')

        iteration+=1

    print(f'\nAverage accuracy: {100 * averageAccuracy}%\n{50*"_"}\n')



# Least Squares Method
print('\n\n', 15 * '_' + 'LEAST_SQUARES_METHOD' + 15 *'_', '\n')
for bc in betCompanies:
    iteration = 1
    averageAccuracy = 0
    for train_index, test_index in kf.split(data[bc.name]):        
        # Changing training and testing sets
        bc.loadData(data[bc.name][train_index], data[bc.name][test_index])
        
        # Calculating weights
        bc.w = LSfit(bc.Xtrain, bc.ytrain)
        
        # Calculating accuracy
        accuracy = bc.calculateAccuracy(bc.predict(bc.Xtest) ,bc.ytest)
        averageAccuracy += accuracy / 10
        print(f'{bc.name} iteration {iteration} | Accuracy: {100 * accuracy}%')

        iteration+=1

    print(f'\nAverage accuracy: {100 * averageAccuracy}%\n{50*"_"}\n')

'''
#K means clustering: TASK 4
print('\n\n', 15 * '_' + 'K_MEANS_CLUSTERING' + 15 *'_', '\n')

cMeans = cMeans()
for bc in betCompanies:
    iteration = 1
    averageAccuracy = 0
    for company in kf.split(data[bc.name]):        
        # Changing training and testing sets
        
        cMeans.cluster()

#Linear Neural Network: TASK 1
print('\n\n', 15 * '_' + 'LINEAR_NEURAL_NETWORK' + 15 *'_', '\n')


network = NeuralNetwork()
for bc in betCompanies[:1]:
    iteration = 1
    averageAccuracy = 0
    # 10fold cross validation
    for train_index, test_index in kf.split(TeamAttributesData):
        print(f'{bc.name} | Linear Neural Network iteration {iteration}\n')

        # Loading data
        network.loadData(TeamAttributesData[train_index], TeamAttributesData[test_index])

        # Calculating weights
        network.fit(iteration,'linear')

        #Calculating accuracy
        accuracy = network.calculateAccuracy(network.predict(network.Xtest,'linear'), network.ytest)
        averageAccuracy += accuracy / 10
        print(f'Accuracy: {100 * accuracy}%\n{50*"_"}\n')

        iteration+=1

        

    print(f'\nAverage accuracy: {100 * averageAccuracy}%\n{50*"_"}\n')

        

#Neural Network
print('\n\n', 15 * '_' + 'NEURAL_NETWORK' + 15 *'_', '\n')
iteration = 1
averageAccuracy = 0
# 10fold cross validation
for train_index, test_index in kf.split(TeamAttributesData):
    # Loading data
    network.loadData(TeamAttributesData[train_index], TeamAttributesData[test_index])
    print(f'Neural Network iteration {iteration}\n')
    iteration += 1

    # Calculating weights
    network.fit()

    #Calculating accuracy
    accuracy = network.calculateAccuracy(network.predict(network.Xtest), network.ytest)
    averageAccuracy += accuracy / 10
    print(f'Accuracy: {100 * accuracy}%\n\n')

print(f'\nAverage accuracy: {100 * averageAccuracy}%\n')