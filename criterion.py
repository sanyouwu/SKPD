# M always means matrix
import numpy as np
import pandas as pd
import os 
import numpy.linalg as la
import copy
# from nonlinear_SKPD import *
import time
from sklearn.model_selection import KFold


def RMSE(pred,truth):
    sample_size = len(truth)
    fn = la.norm(np.asarray(np.asarray(pred).squeeze() - np.asarray(truth).squeeze()), 2) / np.sqrt(sample_size)  # RMSPE from Hongtu ZHU
    return fn

def RMSE_C(C_hat,C):
    ## estimation error in statistics
    d1,d2 = C.shape
    c1 = C_hat.reshape(d1*d2,-1)
    c2 = C.reshape(d1*d2,-1)
    res = la.norm(c1.squeeze()-c2.squeeze(),2) / np.sqrt(d1 *d2)
    return res
def RMSE_tensor(C_hat,C):
    m,n,d = C.shape
    c1 = C_hat.reshape(m*n*d,-1)
    c2 = C.reshape(m*n*d,-1)
    res = la.norm(c1.squeeze()-c2.squeeze(),2) / np.sqrt(m*n*d)
    return res

def error(beta_hat,beta,th = 0):
    ## non-threshold version
    ## return typeI error and Power
    # rmse_c = RMSE_C(beta_hat,beta)
    _len = beta.shape
    if len(_len)== 2:
        rmse_c = RMSE_C(beta_hat,beta)
        m,n = beta.shape
    else:
        rmse_c = RMSE_tensor(beta_hat,beta)
        m,n,d = beta.shape
    beta =  np.where(np.abs(beta) == 0 ,beta,1)
    beta_hat = np.asarray(beta_hat)
    if th == 0:
        tmp = np.where(np.abs(beta_hat) == 0 ,beta_hat,1)
    else:
        tmp = np.where(np.abs(beta_hat) > th,beta_hat,0)
        tmp = np.where(np.abs(tmp) < th,tmp,1)
    diff = beta - tmp
    mul = beta * tmp
    typeI = np.where(diff == -1)[0].shape[0]/np.where(beta == 0)[0].shape[0]
    Power = np.where(mul == 1)[0].shape[0]/np.where(beta == 1)[0].shape[0]
    return typeI,Power,rmse_c
