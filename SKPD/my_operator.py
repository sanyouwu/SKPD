# M always means matrix
import numpy as np
import pandas as pd
import os
# import re
# import string
# import gzip
# from numba import vectorize
import numpy.linalg as la
def Vec(M):
    '''return shape is m*n,1 '''
    return M.reshape(-1,1)
def Vec_inv(M,m,n,d=1):
    # stack direction: axis = 1
    if d == 1:
        return M.reshape(m,n)
    else:
        return M.reshape(m,n,d)
def Rearrange(C,p1,d1,p2,d2,p3=1,d3=1):
    flag = C.shape
    if len(flag) == 2:
        m,n = C.shape
        RC = []
        assert m == p1*d1 and n == p2*d2, "Matrix dimension wrong !!!!!"
        for i in range(p1):
            for j in range(p2):
                Cij = C[d1*i:d1*(i+1),d2*j:d2*(j+1)]
                RC.append(Vec(Cij))
        return np.concatenate(RC,axis = 1).T
    elif len(flag) == 3:
        m,n,d = C.shape
        RC = []
        assert m == p1*d1 and n == p2*d2 and d == p3*d3, "Tensor dimension wrong !!!!"
        for i in range(p1):
            for j in range(p2):
                for k in range(p3):
                    Cij = C[d1*i:d1*(i+1),d2*j:d2*(j+1),d3*k:d3*(k+1)]
                    RC.append(Vec(Cij))
        return np.concatenate(RC,axis = 1).T

def R_inv(RC,p1,d1,p2,d2,p3=1,d3=1):
    if p3 ==1 and d3 ==1:
        p1p2,d1d2 = RC.shape
        C = np.zeros([p1*d1,p2*d2])
        bb = []
    #     print("d1: ",d1)
    #     print("d2: ",d2)

        for i in range(p1p2):

            Block = Vec_inv(RC[i,:],d1,d2)
            ith = i // p2 # quotient
            jth = i % p2  # remainder
            C[d1*ith:d1*(ith+1),d2*jth:d2*(jth+1)] = Block
            bb.append(Block)

        return C,bb
    else:
        p1p2p3,d1d2d3 = RC.shape
        print("RC shape: ",RC.shape)
        C = np.zeros([p1*d1,p2*d2,p3*d3])
        p2p3 = p2*p3 
        bb = []
        print("p2p3",p2*p3)
        for i in range(p1p2p3):
            Block = Vec_inv(RC[i,:],d1,d2,d3)
            # print("i: ",i)
            ith = i //  p2p3 ## quotient
            reminder = i % p2p3  # reminder
            jth = reminder // p3
            kth = reminder % p3
            C[d1*ith:d1*(ith+1),d2*jth:d2*(jth+1),d3*kth:d3*(kth+1)] = Block
            bb.append(Block)
        return C,bb

# for plot gray image 
def convert_to_gray(img):
    m,n = img.shape
    for i in range(m):
        for j in range(n):
            if img[i,j] == 0:
                img[i,j] = 1
            elif img[i,j] ==1 :
                img[i,j] = 0
    return img

# bivariate Haar wavelet basis function
def bi_haar(x):
    if x>= -1 and x < 0:
        x = 1
    elif x >= 0 and x <1 :
        x = -1
    else:
        x = 0
    return x

def RELU(x):
    return np.maximum(0,x)

def sigmoid(x):
    return (1/(1+np.exp(-0.1*x))) * 20

def func_kron_ab(a_hat,b_hat,R,p1,p2,d1,d2,p3=1,d3 = 1):
    Ra_hat = a_hat.reshape(R,-1)
    Rb_hat = b_hat.reshape(R,-1)
    A = []
    B = []
    kron_ab = []
    if p3 == 1 and d3 == 1:
        for i in range(1,R+1):
            locals()['a_hat' + str(i)] = Ra_hat[i-1,:].reshape(-1,1)
            locals()['b_hat' + str(i)] = Rb_hat[i-1,:].reshape(-1,1)
            A.append(Vec_inv(eval('a_hat' + str(i)),p1,p2))
            B.append(Vec_inv(eval('b_hat' + str(i)),d1,d2))

        kron_ab = [np.kron(A[i],B[i]) for i in range(R)]
        beta_hat = sum(kron_ab)
        kron_ab.append(beta_hat)
        
    else:
        Ra_hat = a_hat.reshape(R,-1)
        Rb_hat = b_hat.reshape(R,-1)
        A = []
        B = []
        kron_ab = []
        for i in range(1,R+1):
            locals()['a_hat' + str(i)] = Ra_hat[i-1,:].reshape(-1,1)
            locals()['b_hat' + str(i)] = Rb_hat[i-1,:].reshape(-1,1)
            A.append(Vec_inv(eval('a_hat' + str(i)),p1,p2,p3))
            B.append(Vec_inv(eval('b_hat' + str(i)),d1,d2,d3))

        kron_ab = [np.kron(A[i],B[i]) for i in range(R)]
        beta_hat = sum(kron_ab)
        kron_ab.append(beta_hat)

    return A,B,kron_ab

def fun_th(X):
    ## set  non-zero entries in matrix X to 1
    return np.where(np.abs(X) == 0 ,X,1)

def fun_normalization(data):
    data = np.abs(data)
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def fun_average(data,num = 7):
    ## data is a  list
    new_data = []
    tmp = []
    for i in range(1,len(data)+1,1):
        tmp.append(data[i])
        if i%num ==0:
            new_data.append(np.mean(np.array(tmp),axis = 0))
            tmp = []
    return new_data


def fun_preprocess(Xi):
    fro = la.norm(Xi)
    mu = np.mean(Xi)
    std = np.std(Xi)
    return (Xi-mu)/std

def min_max_norm(Xi):
    _max = np.max(Xi)
    _min = np.min(Xi)
    return (Xi-_min)/(_max-_min)

def Gaussian_lize(C, mu=1, sigma=1):
    idx = np.where(C != 0)
    C[idx[0], idx[1]] = np.random.normal(mu, sigma, len(idx[0]))
    return C