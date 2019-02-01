import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal


data = np.loadtxt('wine.data.txt', delimiter=',')
# Names of features
featurenames = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols',
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline']
# Split 178 instances into training set (trainx, trainy) of size 130 and test set (testx, testy) of size 48
np.random.seed(0)
perm = np.random.permutation(178)
trainx = data[perm[0:130],1:14]
trainy = data[perm[0:130],0]
testx = data[perm[130:178], 1:14]
testy = data[perm[130:178],0]


def fit_generative_model(x,y):
    k = 3  # labels 1,2,...,k
    d = (x.shape)[1]  # number of features
    mu = np.zeros((k+1,d))
    sigma = np.zeros((k+1,d,d))
    pi = np.zeros(k+1)
    for label in range(1,k+1):
        indices = (y == label)
        mu[label] = np.mean(x[indices,:], axis=0)
        sigma[label] = np.cov(x[indices,:], rowvar=0, bias=1)
        pi[label] = float(sum(indices))/float(len(y))
    return mu, sigma, pi


def test_model(mu, sigma, pi, features, tx, ty):
    ###
    k = 3
    mu_select = np.zeros((k + 1, len(features)))
    sigma_select = np.zeros((k + 1, len(features), len(features)))
    for index, feature in enumerate(features):
        for label in range(1, k + 1):
            mu_select[label, index] = mu[label, feature]

    for index1, feature1 in enumerate(features):
        for index2, feature2 in enumerate(features):
            for label in range(1, k + 1):
                sigma_select[label][index1, index2] = sigma[label][feature1, feature2]

    nt = len(ty)
    score = np.zeros((nt, k + 1))
    for i in range(0, nt):
        for label in range(1, k + 1):
            score[i, label] = np.log(pi[label]) + \
                              multivariate_normal.logpdf(tx[i, features], mean=mu_select[label, :],
                                                         cov=sigma_select[label, :, :])
    predictions = np.argmax(score[:, 1:4], axis=1) + 1
    errors = np.sum(predictions != ty)


    return errors


mu, sigma, pi = fit_generative_model(trainx,trainy)
print("The Number of Mistakes Using all Features is",test_model(mu, sigma, pi, list(range(len(featurenames))), testx, testy),"in",len(testy))