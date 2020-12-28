#%%
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import math
from sympy import *
import scipy.stats as stats
from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
from scipy import optimize
#%%
# what if we have two coins (we flip them randomly)?
# Coin_a = ?
# Coin_b = ?
# p_hat(t) = t/n

x = [1,0,0,1,0,1,0,0,0,1,1,1,0,0,1] #flips
z = ['Coin_A']

# Incomplete Information
x = [ (4,6),(3,7),(5,5),(1,9),(7,3),(8,2),(6,4) ] 
# 7 experiments w 10 flips each
# theta always changes, sometimes mean varience, some times 
n = 10
def likelihood(x,theta):
    num_heads, num_tails = x
    return stats.binom.pmf(num_tails,(num_heads+num_tails),theta)
likelihood(x[1],0.9)
# for ex. (3,7); what is probability of getting 7 tails in 10 times 
# in which probability of getting tail is theta/0.9


#%%
def logll(x,p):
    num_heads,num_tails = x
    p_tails = p
    p_heads = 1-p
    return num_tails * np.log(p_tails) + num_heads * np.log(p_heads)

def log_likelihood(x,theta):
    num_heads, num_tails = x
    return stats.binom.logpmf(num_tails,(num_heads+num_tails),theta)
    # to no deal small #s, we are using log
log_likelihood(x[1],0.9)

def neglogll(x,theta):
    num_heads, num_tails = x
    # to no deal w negative #s we multiply it w -1
    return -1 * stats.binom.logpmf(num_tails,(num_heads+num_tails),theta)
#more like to be minimized, more likely to occur
# when neglogll is low, it means more likely to occur. 
neglogll(x[1],0.9)
#%%
from scipy.optimize import minimize

def neglogll(theta,x):
    num_heads, num_tails = x
    return -1 * stats.binom.logpmf(num_tails,(num_heads+num_tails),theta)
#more like to be minimized, more likely to occur
minimize(neglogll, [0.5], args=[1,9], bounds = [(0,1)],method= 'tnc')

# why do we minimize things?
def logll(x,p):
    num_heads,num_tails = x
    p_tails = p
    p_heads = 1-p
    return num_tails * np.log(p_tails) + num_heads * np.log(p_heads)
logll(x[0],0.5)

#%%
minimize(neglogll, [0.5], args=[80,95], bounds = [(0,1)],method= 'tnc')

# there are other methods for optimizing 
# bfgs was problematic so we moved on to tnc
# first numeric libraries, fortan/blast it is conventation is always minimum. 
# people like to see positive numbers more.


#%%
# we find the most likely theta given an experiment
def mle(x):
    num_heads, num_tails = x
    return (num_tails+0.0000000000001) / ((num_heads+num_tails)+0.00000000001)
# we can find how likely an experiment is given theta
def logll(x,p):
    num_heads,num_tails = x
    p_tails = p
    p_heads = 1-p
    return num_tails * np.log(p_tails) + num_heads * np.log(p_heads)
logll(x[0],0.5)
#%%
# ATTENTION TO question
# if lower likelihoood 
# 7 experiments w 10 flips each
xs = [ (4,6),(3,7),(5,5),(1,9),(7,3),(8,2),(6,4)] 
zs = [0,1,0,1,0,0,0]
# thetas = [0.5,0.6] 
# unknown is coin a and coin b`s probabilities , thetas
def find_thetas(xs,zs):
    coin1tails=0
    coin1heads=0
    coin2tails=0
    coin2heads=0
    for x,z in zip(xs,zs):
        num_heads,num_tails = x
        if z == 0:
            coin1tails += num_tails
            coin1heads += num_heads
        else:
            coin2tails += num_tails
            coin2heads += num_heads
    return mle((coin1heads,coin1tails)), mle((coin2heads,coin2tails))
find_thetas(xs,zs)
# What did we do in here?
# We know which xs belongs to which coin; so 
# Sum of their results according to type of coin
# take mle of both of the coin from sum of results you calculated
#%%
# in here, we know probabilty of two coins and result of experiments
# | thetas & xs
# we are gonna try to determine 
# which experiment, xs value, belongs to which type of coin
xs = [ (4,6),(3,7),(5,5),(1,9),(7,3),(8,2),(6,4)] # 7 experiments w 10 flips each
# unkown zs = [0,1,0,1,0,0,0]
thetas = [0.5,0.6] 
# this should be like possible outcomes or sth like that

def logll(x,p):
    num_heads,num_tails = x
    p_tails = p
    p_heads = 1-p
    return num_tails * np.log(p_tails) + num_heads * np.log(p_heads)


def find_zs(xs,thetas):
    theta_a = thetas[0]
    theta_b = thetas[1]
    zs = []
    total_logll = 0
    for x in xs:
        logll_a = logll(x,theta_a) 
        logll_b = logll(x,theta_b)
        # print(x,logll_a,logll_b)
        if logll_a > logll_b:
            zs.append(0)
            total_logll += logll_a
        else:   
            zs.append(1)
            total_logll += logll_b
    return zs, total_logll
    
#%% Expectation Maximization 
# Finding total loggs/zs, record likelihood, reassign everything,
# keep doing this until likelihodd ddoes not improve
def coin_em(cs,initial_guess = [0.1,0.9]): #1
    max_iter = 100 # len(cs) makes more sense
    tol = 0.0001 #6
    thetas = initial_guess
    last_logll = -np.infty
    for i in range(max_iter):
        #E Step    
        zs,total_logll = find_zs(cs,thetas) #2
        # determine which result belongs to which type of coin
        #M Step 
        thetas = find_thetas(cs,zs) #3 
        # determine probability of each of them by looking their mle
        print(f'Iteration {i}:') #4
        print(f' Thetas = {thetas}')
        print(f'Current log likelihood = {total_logll}')
        if total_logll - last_logll < tol: #5
            break
        last_logll = total_logll
    return thetas,zs  
data = [(10-x,x) for x in stats.binom.rvs(10, 0.5, size=100)] + [ (10-x,x) for x in stats.binom.rvs(10,0.2,size=100)]
np.random.shuffle(data)
# what is best initial guess?
#1 start w initial guess for thetas 
#2 use these thetas and assign to one coin and total log
#3 using these assgments, update our parameters, find new probabilties for each coin
#4 report what`s happened 
#5 compare current liklelihood w to the last annd if it less than certain small value ,#6, break
# cuz our likelihood is not improving in that case. 
# our current likelihood becomes last likelihood and we go back and do it again. 
#
#there was sth in first def in whcich 0.5 does sth
#
# for guesses like 0.5 0.5, there is `divison by zero` error; so you need to modify the functions:
# 1.st way:
# adding so small values to mle function like 0.0000001 so there will be no error
# 2.nd way:
#%% 2.nd Way:

def find_ws(xs,thetas):
    theta_a = thetas[0]
    theta_b = thetas[1]
    ws = []
    total_logll = 0
    for x in xs:
        logll_a = logll(x,theta_a) 
        logll_b = logll(x,theta_b)
        # print(x,logll_a,logll_b)
        ll_a = np.exp(logll_a)
        ll_b = np.exp(logll_b)
        denom = ll_a + ll_b
        w = ll_b / denom
        ws.append(w)
        total_logll += np.log((1.0-w) * ll_a) + np.log(w * ll_b)
    # w is going to be somewhere btw 0 & 1
    return ws, total_logll
# using find ws to try to change find thetas to use w instead of z and try to run coin_em
# clustering
# M step ; k means algorithm 
# start w points, start w intitial gguess w centers of different gausians. 
# each iteration moves to most liklely. 
# more compenent added, more complex but likelihood increases; 
# u want to find sweet find like good likelihood but still simple model.
# gaussian mixture modelling

#%% Trying for doing same thing find_zs with ng; however, there are some missing points. 
# u can try 
def neglogll(x,theta):
    num_heads, num_tails = x
    # to no deal w negative #s we multiply it w -1
    return -1 * stats.binom.logpmf(num_tails,(num_heads+num_tails),theta)
#more like to be minimized, more likely to occur
# when neglogll is low, it means more likely to occur. 
neglogll(x[1],0.9)

def find_zsn(xs,thetas):
    theta_a = thetas[0]
    theta_b = thetas[1]
    zs = []
    total_logll = 0
    for x in xs:
        nlogll_a = neglogll(x,theta_a)
        nlogll_b = neglogll(x,theta_b)
        # print(x,nlogll_a,nlogll_b)
        if nlogll_a < nlogll_b:
            zs.append(0)
            total_logll += nlogll_a
        else:
            zs.append(1)
            total_logll += nlogll_b
    return zs, total_logll
    
# %%
from numpy.random import normal
import matplotlib.pyplot as plt 
sample1 = np.normal(loc = 20, scale = 5, size = 400) 
sample1 = np.normal(loc = 40, scale = 5, size = 800) 
sample = sample1 + sample2
plt.hist(sample,bin=50,density=True) 
