import numpy as np
import pandas as pd

def therm_lower(n):
    return np.vstack([np.tril(np.ones((n,n))),np.zeros(n)])
def therm_upper(n):
    return np.vstack([np.triu(np.ones((n,n))),np.zeros(n)])
def onehot(n):
    return np.eye(n)
def maxonehot(n):
    return np.vstack([np.eye(n),np.zeros(n)])
def num_range(l,u):
        return [np.array([i]) for i in range(l,u+1)]
