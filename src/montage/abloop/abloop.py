#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""abloop.py: Skeleton operations of the abductive loop."""

__author__      = "Brian J. Goode"

import pandas as pd
import numpy as np
from scipy.stats import chi2

from sklearn.metrics import mutual_info_score
import seaborn as sns
from matplotlib import pyplot as plt

class Abloop:
    
    hypotheses = {}
    treatments = {}
    
    def __init__(self):
        
        return
    
    
    def import_csv(self,csv,idx):
        """import_csv: import experimental data into the abductive loop"""
        
        self.data = pd.read_csv(csv, index_col=idx)
        return
    
    
    def print_features(self):
        cols = self.data.columns
        print("Available Features:")
        print("    " + "\n    ".join(cols))
        return
    
    
    def set_prior(self):
        """set_prior: set prior beliefs in the abductive loop"""
        ## For future addition - have only for objective priors now...
        
        pass
    
    
    def add_hypothesis(self, inFeatures, depFeatures, model, treatment = None, label = None):
        
        if not label:
            label = len(self.hypotheses)
        
        hypothesis = {
            'x': inFeatures,
            'y': depFeatures,
            'model': model,
            'treatment': treatment,
        }
        
        self.hypotheses[label] = hypothesis
        return
    
    
    def add_treatment(self, session_list, label = None):
        """set_treatment: set treatments for experiment"""
        
        if not label:
            label = len(self.treatments)
        
        # Need to enforce criteria on session_list input...
        # For now, it should literally be a list of sessions; future should be
        # a dict of some design.
        
        self.treatments[label] = session_list
        return
    
    
    def plot_treatment_distribution(self, h, xList, zList):
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        i = 0
        legs = []
        for t2 in zList:
            means = []
            stds = []

            for t1 in xList:
                x,y = self.prepareData(h,[t1],t2)
            
                means.append(y.mean())
                stds.append(y.std()*2.)
            
            plt.errorbar(range(len(means)),means,yerr=stds, capsize = 10,marker='o', markersize=10, elinewidth=4, capthick=4, linewidth=4, label=t2)
            legs.append(plt.Rectangle((0, 0), 1, 1, fc=sns.color_palette()[i]))
            i += 1

        plt.xticks(range(len(xList)),xList, fontsize=12)
        plt.legend(legs, zList, frameon=False, fontsize = 12)

        return
    
    
    def prepareData(self, h, treatment_list=None, *args):
        # Add in feature for train/test delineation...
        
        data = self.data
        
        if treatment_list:
            session_list = []
            for t in treatment_list:
                session_list.extend(self.treatments[t])
            data = self.data.loc[session_list]
            
        for a in args:
            session_list = self.treatments[a]
            data = data[data.index.isin(session_list)]
            
        
        D = data[h['x'] + h['y']].dropna()    
        x = D[h['x']].as_matrix()
        y = D[h['y']].as_matrix()
        
        return x,y
    
    
    def estimate(self):
        
        for label, h in self.hypotheses.iteritems():
            x,y = self.prepareData(h)
            _ = h['model'].train(x,y)
        
        return
    
    
    def estimate_predicted_effect_size(self, x0, h0, x1, h1):
        
        """ Estimate effect size between h0, h1. """
        
        # Standard deviation calculator between outputs h1, h2 for treatments...
        # Plot below...
        
        y0 = h0['model'].sample(x)
        m0 = y0.mean()
        sd0 = y0.std()
        
        y1 = h1['model'].sample(x)
        m1 = y1.mean()
        sd1 = y1.std()
        
        def effect_size(m0,m1,sd0,sd1):
            num = m1-m0
            den = ((sd0**2 + sd1**2)*0.5)**0.5
            return num/den
        
        return effect_size(m0,m1,sd0,sd1)
    
    
    def estimate_treatment_effect_size(self, h, t0, t1):
        
        """ Estimate effect size between h1, h2. """
        
        # Standard deviation calculator between outputs h1, h2 for treatments...
        # Plot below...
        
        x,y0 = self.prepareData(h,[t0])
        x,y1 = self.prepareData(h,[t1])
        
        m0 = y0.mean()
        m1 = y1.mean()
        sd0 = y0.std()
        sd1 = y1.std()
        
        def effect_size(m0,m1,sd0,sd1):
            num = m1-m0
            den = ((sd0**2 + sd1**2)*0.5)**0.5
            return num/den
        
        return effect_size(m0,m1,sd0,sd1)
    
    
    def print_estimates(self, hypothesis = None):
        
        # FIX: Make sure prints right (handle features)
        
        hypotheses = self.hypotheses
        
        if hypothesis is not None:
            hypotheses = dict(hypothesis, self.hypotheses[hypothesis])
        
        for label, h in hypotheses.iteritems():
            parameters = h['model'].getParameterEstimate()
            
            print('Hypothesis {}'.format(label))
            
            for pk,pv in parameters.iteritems():
                print('\n'.join(['{:40}{}'.format(*x) for x in zip(h['x'],pv)]))
                
        return
    
    
    def plot_estimates(self, hypothesis = None):
        # NOT DONE YET!     
        # Plot estimates here.
        """for bs, title in zip(bsamples.T,titles):
            plt.figure()
            _ = plt.hist(bs, 1000, normed = True)
            plt.title(title)
            plt.xlabel(r'$\beta$')
            plt.ylabel(r'$P(\beta|y)$')"""
            
        return
    
    
    def calcMseErrDist(self, y_hat, y):
        err_samples = y_hat.T - y
        err_bar =  (err_samples**2).mean(axis = 0)
        dof = len(err_samples)

        loc, scale =  chi2.fit_loc_scale(err_bar, dof)
        err_dist = chi2(dof, loc=loc, scale=scale)
        
        return err_dist
    
    
    def validate(self, hypothesis = None):
        """validate: validate model on out-of-sample data"""
        
        hypotheses = self.hypotheses
        
        if hypothesis is not None:
            hypotheses = dict(hypothesis, self.hypotheses[hypothesis])
        
        for label, h in hypotheses.iteritems():
            x,y = self.prepareData(h)
            
            y_hat = h['model'].sample(x)
            h['err_dist'] = self.calcMseErrDist(y_hat,y)
            
        return
    
    
    def abduce_hypotheses(self, xaxis = None):
        
        if not xaxis:
            xaxis = np.logspace(0,19,1000,base=2)
        
        hypotheses = self.hypotheses
        
        P_he = []
        for label, h in hypotheses.iteritems():
            P_he.append(h['err_dist'].pdf(xaxis))
        
        P_he_marginal = np.array(P_he)
        P_he = np.array(P_he)
        den = P_he.sum(axis=0)
        P_he = P_he/den

        self.xaxis = xaxis
        self.P_he = P_he
        self.P_he_marginal = P_he_marginal
        return
    
    
    def plot_abduce_hypotheses(self):
        
        xaxis = self.xaxis
        
        for phe, label in zip(self.P_he, self.hypotheses):
            plt.plot(xaxis,phe, label=label)
        
        plt.semilogx(basex = 10)
        plt.xlabel('MSE', fontsize = 16)
        _ = plt.ylabel(r'$P(h|\tilde e)$', fontsize = 16)
        plt.legend()
        return
    
    
    def plot_abduce_hypotheses_marginal(self):
        
        xaxis = self.xaxis
        
        for phe, label in zip(self.P_he_marginal, self.hypotheses):
            plt.plot(xaxis,phe, label=label)
        
        plt.semilogx(basex = 10)

        plt.xlabel('MSE', fontsize = 16)
        _ = plt.ylabel(r'$P(h,\tilde e|H)$', fontsize = 16)
        plt.legend()
        
        return
    
    
    def set_input_output(self, invar, outvar):
        
        self.invar = invar
        self.outvar = outvar
        
        return
    
    
    def abduce_results(self, depVarName):
        features = self.data.columns
        n_features = len(features)
        data = self.data

        MI = np.empty(shape = (n_features,n_features))
        for i in range(n_features):
            for j in range(n_features):
                Cxy = np.histogram2d(data[features[i]].replace(np.nan,0),data[features[j]].replace(np.nan,0))[0]
                MI[i,j] = mutual_info_score(None,None,contingency=Cxy)
        
        MI = pd.DataFrame(MI,columns = data.columns, index = data.columns)
        results = MI[depVarName].loc[self.invar].sort_values(ascending=False)
        
        return results
    
    
    def load(self):
        pass
    
    
    def save(self):
        pass