# M always means matrix
import numpy as np
import pandas as pd
import os
import re
import string
import gzip
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
def Rearrange(C,m1,m2,n1,n2,d1=1,d2=1):
    flag = C.shape
    if len(flag) == 2:
        m,n = C.shape
        RC = []
        assert m == m1*m2 and n == n1*n2, "Matrix dimension wrong !!!!!"
        for i in range(m1):
            for j in range(n1):
                Cij = C[m2*i:m2*(i+1),n2*j:n2*(j+1)]
                RC.append(Vec(Cij))
        return np.concatenate(RC,axis = 1).T
    elif len(flag) == 3:
        m,n,d = C.shape
        RC = []
        assert m == m1*m2 and n == n1*n2 and d == d1*d2, "Tensor dimension wrong !!!!"
        for i in range(m1):
            for j in range(n1):
                for k in range(d1):
                    Cij = C[m2*i:m2*(i+1),n2*j:n2*(j+1),d2*k:d2*(k+1)]
                    RC.append(Vec(Cij))
        return np.concatenate(RC,axis = 1).T

def R_inv(RC,m1,m2,n1,n2,d1=1,d2=1):
    if d1 ==1 and d2 ==1:
        m1n1,m2n2 = RC.shape
        C = np.zeros([m1*m2,n1*n2])
        bb = []
    #     print("m2: ",m2)
    #     print("n2: ",n2)

        for i in range(m1n1):

            Block = Vec_inv(RC[i,:],m2,n2)
            ith = i // n1 # quotient
            jth = i % n1  # remainder
            C[m2*ith:m2*(ith+1),n2*jth:n2*(jth+1)] = Block
            bb.append(Block)

        return C,bb
    else:
        m1n1d1,m2n2d2 = RC.shape
        print("RC shape: ",RC.shape)
        C = np.zeros([m1*m2,n1*n2,d1*d2])
        n1d1 = n1*d1 
        bb = []
        print("n1d1",n1*d1)
        for i in range(m1n1d1):
            Block = Vec_inv(RC[i,:],m2,n2,d2)
            # print("i: ",i)
            ith = i //  n1d1 ## quotient
            reminder = i % n1d1  # reminder
            jth = reminder // d1
            kth = reminder % d1
            C[m2*ith:m2*(ith+1),n2*jth:n2*(jth+1),d2*kth:d2*(kth+1)] = Block
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
def linear(x):
    return x
def square(x):
	return 1/20 * x**2

def neg_relu(x):
    return np.minimum(0,x)

def sigmoid(x):
    return (1/(1+np.exp(-0.1*x))) * 20

def soft_th(X,tau = 15):
    return np.sign(X) * np.maximum((np.abs(X) - tau), np.zeros(X.shape))

def func_kron_ab(a_hat,b_hat,R,m1,n1,m2,n2,d1=1,d2 = 1):
    Ra_hat = a_hat.reshape(R,-1)
    Rb_hat = b_hat.reshape(R,-1)
    A = []
    B = []
    kron_ab = []
    if d1 == 1 and d2 == 1:
        for i in range(1,R+1):
            locals()['a_hat' + str(i)] = Ra_hat[i-1,:].reshape(-1,1)
            locals()['b_hat' + str(i)] = Rb_hat[i-1,:].reshape(-1,1)
            A.append(Vec_inv(eval('a_hat' + str(i)),m1,n1))
            B.append(Vec_inv(eval('b_hat' + str(i)),m2,n2))

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
            A.append(Vec_inv(eval('a_hat' + str(i)),m1,n1,d1))
            B.append(Vec_inv(eval('b_hat' + str(i)),m2,n2,d2))

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

def fun_preprocess(data):
    new_data = []
    for Xi in data:
        fro = la.norm(Xi)
        new_data.append(Xi/fro)
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