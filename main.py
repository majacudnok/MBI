import numpy as np
import csv
from sklearn.metrics import mean_squared_error
import time
import os

if  __name__ =='__main__':
   print('Start:')
import sys;
print(sys.maxsize)
global_start = time.time()

# prepare first matrix with id and ethnic group
dir = os.path.dirname(__file__)
filename = os.path.join(dir,'out.txt')

idToEthnicGroup = np.genfromtxt(filename, dtype='str',delimiter='\n',usecols=(0),
                                skip_header=1) # usecols - we only take 1st and 3rd column, skip header skips first row with row headers

# prepare second matrix with samples and snps
filename2 = os.path.join(dir, 'chr22qc_example.csv')
if os.path.exists(filename2):
    matrix = []
    with open(filename2, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            matrix.append(row)
        f.close()

    print(matrix.__len__())
    testCasesNo = len(matrix)
    dt = np.dtype('b')
    dataMatrix = np.array(matrix[:testCasesNo])
    idToEthnicGroup = (np.array(dataMatrix[0:,1]))
    outputData = np.empty([testCasesNo,1])
    i = 0
    for ethnicGroup in idToEthnicGroup:
        if ethnicGroup == 'EUR':
            outputData[i,0] = 0.2
            i += 1
        elif ethnicGroup == 'SAS':
            outputData[i, 0] = 0.4
            i += 1
        elif ethnicGroup == 'AMR':
            outputData[i, 0] = 0.6
            i += 1
        elif ethnicGroup == 'AFR':
            outputData[i, 0] = 0.8
            i += 1
        elif ethnicGroup == 'EAS':
            outputData[i, 0] = 1
            i += 1

    inputData = [row[1:] for row in matrix[:testCasesNo]]
    print('done')

    # input parameters:
    y = outputData[200:500,:]
    X = np.array(dataMatrix[200:500,2:1000])

    # X = np.array([[0, 0, 1],
    #               [0, 1, 1],
    #               [1, 0, 1],
    #               [1, 1, 1],
    #               [2, 2, 2],
    #               [3, 3, 0],
    #               [3, 3, 1],
    #               [5, 2, 2]])
    #
    # y = np.array([[1],
    #               [1],
    #               [1],
    #               [1],
    #               [0],
    #               [0.5],
    #               [0.5],
    #               [0.8]])


######################

    # sigmoid function
    def nonlin(x, deriv=False):
        if (deriv == True):
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # neuron network initialization

    def networkInitialization(neuronsIn, neuronsHide, neuronsOut):
        # np.random.seed(1)
        syn0 = 0.3 * np.random.random((neuronsIn, neuronsHide)) - 0.15
        syn1 = 0.3 * np.random.random((neuronsHide, neuronsOut)) - 0.15

        return syn0,syn1


    def networkLearning (syn0,syn1, epocs):
        for j in range(epocs):
            print('start iteration', j)

            # through layers 0, 1, and 2
            l0 = np.float32(X)

            l1 = nonlin(np.dot(l0, syn0))
            l2 = nonlin(np.dot(l1, syn1))

            # error correction
            l2_error = y - l2

            if (j % 10000) == 0:
                print
                "Error:" + str(np.mean(np.abs(l2_error)))

            l2_delta = l2_error * nonlin(l2, deriv = True)

            l1_error = l2_delta.dot(syn1.T)

            l1_delta = l1_error * nonlin(l1, deriv = True)

            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)

        return l2, l2_error

    def evaluate(y,l2):
        from sklearn.metrics import r2_score;
        MSE = mean_squared_error(y, l2[:, 0])
        print(MSE)

        r2_score = r2_score(y, l2[:,0])
        return r2_score, MSE

    [syn0, syn1] = networkInitialization(X[0].size, 10, 5)
    [l2, l2_error] = networkLearning(syn0, syn1, 50)
    [r2_score, MSE] = evaluate(y,l2)

    print(l2)
    print(l2_error)
    ################3

else:
    print("File not found")
