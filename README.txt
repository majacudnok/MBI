MBI - neuronal network for assigning to ethnic groups based on SNPs

Project made by Kaja Etmanowicz and Maja Cudnok

Input parameters:
example:
     # input parameters:
 58.   y = outputData[200:500,:]
 59.   X = np.array(dataMatrix[200:500,2:1000])

y - samples
X - samples x SNP

Network parameters:

 134.  [syn0, syn1] = networkInitialization(X[0].size, 10, 5)
 135.   [l2, l2_error] = networkLearning(syn0, syn1, 50)

Where:
10 - number of hidden neurons
5 - number of outputs
syn0, syn1 - weights matrixes initialized in function networkInizalization
50 - number of iterations