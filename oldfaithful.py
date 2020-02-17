# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
oldfaithful = pd.read_csv("oldfaithful.csv")

#initialize parameters
k = 2
mu = np.array([[2.0,40.0],[3.0,50.0]])
cov = np.array([[[3,0],[0,3]],[[1,0],[0,1]]])
pi = np.array([0.3,0.7])
#convert data to an array so it is easier to use in the EM algorithm
x = oldfaithful.to_numpy()
#length of data set
length = len(x)
#used to store values from E step
gamma = [[0 for j in range(length)]for i in range(k)]


#EML algorithm implementation   
for s in range(30):
    #E step
    #loop through all the points
    for n in range(length):
        for i in range(k):
            #numerator of the expectation
            numerator = pi[i] * stats.multivariate_normal.pdf(x[n], mu[i], cov[i])
            #sum up all the values to get the denominator
            denominator = 0
            for j in range(k):
                denominator += pi[j] * stats.multivariate_normal.pdf(x[n], mu[j], cov[j])
            #expectation for a specific point in a guassian, k
            gamma[i][n] = numerator / denominator

    #M step
    #loop through the number of gaussians
    for i in range(k):
        n_k = 0
        for n in range(length):
            n_k += gamma[i][n]
        #assign new value of pi
        pi[i] = n_k / length
        #loop through points and get the numerator for new mean value
        mu_numerator = 0
        for n in range(length):
            mu_numerator += (gamma[i][n] * x[n])
        #assign the new mean vallue using the numerator
        mu[i] = mu_numerator / n_k
        #loop through the points and get the numerator for the new covalence matrix
        cov_numerator = 0
        for n in range(length):
            cov_numerator += (gamma[i][n] * np.dot((x[n]-mu[i]),np.transpose(x[n]-mu[i])))
        #assign new covarinace value so that there is not a singular matrix
        cov[i][0][0] = cov_numerator / n_k
        cov[i][1][1] = cov_numerator / n_k
        cov[i][0][1] = 0
        cov[i][1][0] = 0


#maximizing log likelihood
log_likelihood_arr = [0 for i in range(length)]
for n in range(length):
    tot = 0
    for i in range(k):
        tot += pi[i] * stats.multivariate_normal.pdf(x[n], mu[i], cov[i])
    log_likelihood_arr[n] = np.log(tot)
    
    
#print(log_likelihood_arr)
#save marginal probabilities in the array to be used to shade the points
marginal_prob_arr = [0 for i in range(length)]
for n in range(length):
    marginal_prob = 0
    for i in range(k):
        marginal_prob += pi[i] * stats.multivariate_normal.pdf(x[n], mu[i], cov[i])
        marginal_prob_arr[n] = marginal_prob
        
 
#plot old faithful data withe color being the marginal probabilities
ax = oldfaithful.plot.scatter(x = "eruptions", y = "waiting", c = marginal_prob_arr)
point = pd.DataFrame({"eruptions": [mu[0][0], mu[1][0]], "waiting": [mu[0][1], mu[1][1]]})
#plot means found using EM algorithm
ax = point.plot.scatter(x = "eruptions", y = "waiting", ax = ax, c = "red")

#printing results
print("mean 1:", mu[0])
print("mean 2:", mu[1])
print("cov 1:", cov[0])
print("cov 2:", cov[1])  


#repating the steps for k = 3
#initialize parameters
k = 3
mu = np.array([[3.0,40.0],[4.0,70.0], [3.0, 50.0]])
cov = np.array([[[3,0],[0,3]],[[1,0],[0,1]], [[2,0],[0,2]]])
pi = np.array([0.3,0.5, 0.2])
#convert data to an array so it is easier to use in the EM algorithm
x = oldfaithful.to_numpy()
#length of data set
length = len(x)
#used to store values from E step
gamma = [[0 for j in range(length)]for i in range(k)]


#EML algorithm implementation   
for s in range(30):
    #E step
    #loop through all the points
    for n in range(length):
        for i in range(k):
            #numerator of the expectation
            numerator = pi[i] * stats.multivariate_normal.pdf(x[n], mu[i], cov[i])
            #sum up all the values to get the denominator
            denominator = 0
            for j in range(k):
                denominator += pi[j] * stats.multivariate_normal.pdf(x[n], mu[j], cov[j])
            #expectation for a specific point in a guassian, k
            gamma[i][n] = numerator / denominator

    #M step
    #loop through the number of gaussians
    for i in range(k):
        n_k = 0
        for n in range(length):
            n_k += gamma[i][n]
        #assign new value of pi
        pi[i] = n_k / length
        #loop through points and get the numerator for new mean value
        mu_numerator = 0
        for n in range(length):
            mu_numerator += (gamma[i][n] * x[n])
        #assign the new mean vallue using the numerator
        mu[i] = mu_numerator / n_k
        #loop through the points and get the numerator for the new covalence matrix
        cov_numerator = 0
        for n in range(length):
            cov_numerator += (gamma[i][n] * np.dot((x[n]-mu[i]),np.transpose(x[n]-mu[i])))
        #assign new covarinace value so that there is not a singular matrix
        cov[i][0][0] = cov_numerator / n_k
        cov[i][1][1] = cov_numerator / n_k
        cov[i][0][1] = 0
        cov[i][1][0] = 0


#maximizing log likelihood
log_likelihood_arr = [0 for i in range(length)]
for n in range(length):
    tot = 0
    for i in range(k):
        tot += pi[i] * stats.multivariate_normal.pdf(x[n], mu[i], cov[i])
    log_likelihood_arr[n] = np.log(tot)
    
    
#print(log_likelihood_arr)
#save marginal probabilities in the array to be used to shade the points
marginal_prob_arr = [0 for i in range(length)]
for n in range(length):
    marginal_prob = 0
    for i in range(k):
        marginal_prob += pi[i] * stats.multivariate_normal.pdf(x[n], mu[i], cov[i])
        marginal_prob_arr[n] = marginal_prob
        
 
#plot old faithful data withe color being the marginal probabilities
ax = oldfaithful.plot.scatter(x = "eruptions", y = "waiting", c = marginal_prob_arr)
# =============================================================================
# point = pd.DataFrame({"eruptions": [mu[0][0], mu[1][0], mu[2][0]], "waiting": [mu[0][1], mu[1][1], mu[2][1]]})
# =============================================================================
#plot means found using EM algorithm
ax = point.plot.scatter(x = "eruptions", y = "waiting", ax = ax, c = "red")

#printing results
print("mean 1:", mu[0])
print("mean 2:", mu[1])
print("mean 3:", mu[2])
print("cov 1:", cov[0])
print("cov 2:", cov[1])  
print("cov 3:", cov[2])

