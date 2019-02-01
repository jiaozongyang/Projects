import numpy as np

# Load data set and code labels as 0 = ’NO’, 1 = ’DH’, 2 = ’SL’
labels = [b'NO', b'DH', b'SL']
data = np.loadtxt('column_3C.dat', converters={6: lambda s: labels.index(s)} )

# Separate features from labels
x = data[:,0:6]
y = data[:,6]

# Divide into training and test set
training_indices = list(range(0,20)) + list(range(40,188)) + list(range(230,310))
test_indices = list(range(20,40)) + list(range(188,230))

trainx = x[training_indices,:]
trainy = y[training_indices]
testx = x[test_indices,:]
testy = y[test_indices]


def NN_L2(trainx, trainy, testx):
    # inputs: trainx, trainy, testx <-- as defined above
    # output: an np.array of the predicted values for testy

    distances = []
    for i in range(len(testx)):
        temp = []
        for j in range(len(trainx)):
            distance = np.sum(np.square(testx[i] - trainx[j]))
            temp.append(distance)
        distances.append(np.argmin(temp))

    result = np.array([trainy[i] for i in distances])

    return result


def NN_L1(trainx, trainy, testx):
    # inputs: trainx, trainy, testx <-- as defined above
    # output: an np.array of the predicted values for testy

    distances = []
    for i in range(len(testx)):
        temp = []
        for j in range(len(trainx)):
            distance = np.sum(np.abs(testx[i] - trainx[j]))
            temp.append(distance)
        distances.append(np.argmin(temp))

    result = np.array([trainy[i] for i in distances])

    return result



def error_rate(testy, testy_fit):
    return float(sum(testy!=testy_fit))/len(testy)


def confusion(testy, testy_fit):
    # inputs: the correct labels, the fitted NN labels
    # output: a 3x3 np.array representing the confusion matrix as above

    ### BEGIN SOLUTION
    result = np.array([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    for i in range(len(testy)):

        if testy[i] == 0 and testy_fit[i] == 1:
            result[0][1] += 1
        elif testy[i] == 0 and testy_fit[i] == 2:
            result[0][2] += 1
        elif testy[i] == 1 and testy_fit[i] == 0:
            result[1][0] += 1
        elif testy[i] == 1 and testy_fit[i] == 2:
            result[1][2] += 1
        elif testy[i] == 2 and testy_fit[i] == 0:
            result[2][0] += 1
        elif testy[i] == 2 and testy_fit[i] == 1:
            result[2][1] += 1

    return result



testy_L1 = NN_L1(trainx, trainy, testx)
testy_L2 = NN_L2(trainx, trainy, testx)
L1_neo = confusion(testy, testy_L1)
L2_neo = confusion(testy, testy_L2)
print("Error rate of NN_L1: ", error_rate(testy,testy_L1) )
print("Error rate of NN_L2: ", error_rate(testy,testy_L2) )
print("Confusion Matrix of NN_L1:", L1_neo)
print("Confusion Matrix of NN_L2:", L2_neo)