#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""abloop.py: Skeleton operations of the abductive loop."""

__author__      = "Brian J. Goode"


import numpy as np
from scipy.stats import invgamma, norm, multivariate_normal
from scipy.stats import zscore

class LinearBayesian:      
    
    def __init__(self):
        
        self.n_samples = 10000
        
        pass

    
    def loadTrainingData(self, x, y):
        
        x = zscore(x, axis = 0)
        x = np.c_[x,np.ones([len(x),1])]
        
        self.xTrain = x
        self.yTrain = y
        
        self.n = float(len(y)) ## import from data
        self.p = float(x.shape[1])   ## import from data

        self.inv = np.linalg.inv(np.dot(x.T,x)) 
        self.P = np.dot(np.dot(x,self.inv),x.T)
        
        return
    
    
    def estimateVariance(self):
        """ Estimate variance of entire system."""
        
        x = self.xTrain
        y = self.yTrain
        
        n = self.n
        p = self.p   ## import from data

        inv = self.inv 
        P = self.P
        
        s2hat = (1./(n-p))*np.dot(np.dot(y.T,np.eye(P.shape[0])-P), y)
        
        self.s2hat = s2hat
        
        return
    
    
    def estimateCoefficients(self):
        """ Estimate coefficients using training data. Inputs: x; Outputs: y """
        x = self.xTrain
        y = self.yTrain
        
        inv = self.inv

        Bhat = np.dot(inv,np.dot(x.T,y))
        self.Bhat = Bhat
        
        return
    
    
    def getSampleCoefficients(self):
        
        Bhat = self.Bhat
        s2samples = self.s2samples
        
        inv = self.inv
        
        bsamples = []
        for s in s2samples:
            b = multivariate_normal.rvs(mean = np.ravel(Bhat), cov = np.dot(inv,s))
            bsamples.append(b)
    
        self.bsamples = np.array(bsamples)
        return bsamples
    
    
    def getSampleVariance(self):
        n = self.n
        p = self.p
        
        n_samples = self.n_samples
        
        a = (n-p)/2.
        s2hat = self.s2hat
        scale = ((n-p) * s2hat / 2.)
        
        self.s2samples = invgamma.rvs(a, scale=scale, size=n_samples)
        return self.s2samples
    
    
    def predict(self, x):
        
        x = zscore(x, axis = 0)
        x = np.c_[x,np.ones([len(x),1])]
        
        bsamples = self.bsamples
        sigma2hat = self.s2samples
        
        Y_hat = []
        for b,s in zip(bsamples,sigma2hat):
            y_hat = norm.rvs(np.dot(x,b),s)
            Y_hat.append(y_hat)
        
        self.Y_hat = np.array(Y_hat)
        
        return self.Y_hat
    
    
    def getParameterEstimate(self):
        parameters = {
            'Bhat': self.Bhat,
            's2hat': self.s2hat,
        }
        
        return parameters
    
    
    def getParameterSample(self):
        samples = {
            'B': self.bsamples,
            's2': self.s2samples,
        }
        
        return samples

    
    def train(self, xTrain, yTrain):
        _ = self.loadTrainingData(xTrain,yTrain)
        _ = self.estimateVariance()
        _ = self.estimateCoefficients()
        
        return self.getParameterEstimate()
    
    
    def sampleParameters(self, n_samples = None):
        
        if n_samples is not None:
            self.n_samples = n_samples
        
        _ = self.getSampleVariance()
        _ = self.getSampleCoefficients()
    
        return self.getParameterSample()
    
    
    def sample(self, x, n_samples = None):
        
        _ = self.sampleParameters(n_samples)
        _ = self.predict(x)
        
        return self.Y_hat