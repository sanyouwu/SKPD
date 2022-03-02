    # M always means matrix
import numpy as np
import pandas as pd
import os 
import numpy.linalg as la
import copy
from SKPD import *
import time
from sklearn.model_selection import KFold
from joblib import Parallel,delayed
from .criterion import * 


### update 2021.06.23
### for parallel computing
def skpdRegressor(p1_list,p2_list,d1_list,d2_list,p3_list,d3_list,lmbda_set,lmbda2_set,Z_train,X_train,Y_train,R_list,n_cores = -1,max_iter = 20,print_iter = 5):
    na = True
    nb = True
    ## one simulation with fixed dimension and lambda
    if len(lmbda_set) == 1 and len(p1_list) == 1 and len(R_list) == 1: 
        p1,p2 = p1_list[0],p2_list[0] ### p1,p2
        d1,d2 = d1_list[0],d2_list[0]
        p3,d3 = p3_list[0],d3_list[0]
        R = R_list[0]
        RX = [Rearrange(xi,p1,d1,p2,d2,p3,d3) for xi in X_train]
        opt_solver = AltMin(Z_train,RX,Y_train,lmbda_set[0],lmbda2_set[0],p1,p2,d1,d2,p3,d3,R,na,nb)
        a_hat,b_hat,gamma_hat,Y_hat,err_beta,fN = opt_solver.fit(max_iter = max_iter,iter_print = print_iter)
        return a_hat,b_hat,gamma_hat,lmbda_set[0],lmbda2_set[0],R,p1,p2,p3,d1,d2,d3
    ## No parallel
    if n_cores == None:
        val_values = fun_validate(p1_list,p2_list,d1_list,d2_list,p3_list,d3_list,lmbda_set,lmbda2_set,Z_train,X_train,Y_train,R_list,na,nb,max_iter,print_iter)
    # we take Modified BIC to select lambda in SKPD
    else:
        ## start parallel computing
        print("-------start parallel computing-----------")
        Parameters = pack_parameters(Z_train,X_train,Y_train,lmbda_set,lmbda2_set,p1_list,p2_list,d1_list,d2_list,p3_list,d3_list,R_list,na,nb)
        parallel_res = my_parallel(Parameters,n_cores)  ## return 4 x 5 = 20 array
        val_values = list(np.array(parallel_res).squeeze())

### print results
    # print("Sparsity levels: ",S)
    print("MBIC values: ",val_values)
    # print("\n")
    opt_idx = val_values.index(min(val_values))
    # print("opt-idx: ",opt_idx)
    fir_idx = opt_idx//(len(R_list) * len(lmbda_set))
    tmp1 = opt_idx % (len(R_list) * len(lmbda_set))
    sec_idx = tmp1// len(lmbda_set)
    third_idx = tmp1 % len(lmbda_set)
    # print("fir_idx: ",fir_idx)
    p1,p2,p3 = p1_list[fir_idx],p2_list[fir_idx],p3_list[fir_idx]
    d1,d2,d3 = d1_list[fir_idx],d2_list[fir_idx],d3_list[fir_idx]
    # print("d1: ",d1)
    # print("d2: ",d2)
    # print("d3: ",d3)
    R = R_list[sec_idx]
    # print("R: ",R)
    lmbda_1 = lmbda_set[third_idx]
    lmbda_2 = lmbda2_set[0]
    # opt_solver = solver_list[opt_idx]

   ### output the final estimations and model
    # gamma_hat = opt_solver.betak
    RX = [Rearrange(xi,p1,d1,p2,d2,p3,d3) for xi in X_train]
#     RX_val = [Rearrange(xi,p1,p3,p2,d3,p3,d3) for xi in X_val]
#     RX.extend(RX_val)
#     Y_train.extend(Y_val)
    opt_solver = AltMin(Z_train,RX,Y_train,lmbda_1,lmbda_2,p1,p2,d1,d2,p3,d3,R,na,nb)
    a_hat,b_hat,gamma_hat,Y_hat,err_beta,fN = opt_solver.fit(max_iter = max_iter,iter_print = print_iter)
    # return a_hat,b_hat,gamma_hat,0,0,fN,opt_solver,p1,p2,p3,d3,p3,d3,lmbda_1,lmbda_2,min(val_values)
    return a_hat,b_hat,gamma_hat,lmbda_1,lmbda_2,R,p1,p2,p3,d1,d2,d3



def my_parallel(Parameters,n_cores):
    return Parallel(n_jobs= n_cores, verbose=False)(delayed(my_parallel_lmbda)(*Parameters[i]) for i in range(len(Parameters)))

def my_parallel_lmbda(Z_train,RX_train,Y_train,lmbda1,lmbda2,p1,p2,d1,d2,p3,d3,R,na,nb,max_iter = 20,print_iter = 5):
    ##### start training
#     RX_train.extend(RX_val)
    solver = AltMin(Z_train,RX_train,Y_train,lmbda1,lmbda2,p1,p2,d1,d2,p3,d3,R,na,nb)
    a_hat,b_hat,gamma_hat,Y_hat,err_beta,fN = solver.fit(max_iter = max_iter,iter_print = print_iter)
    # sample_size = len(Y_train)
    if fN == np.Inf:
        # kf_cv.append(np.Inf)
        return np.Inf
    else:
        val_y = solver.fun_predict(Z_train,solver.RX,a_hat,b_hat,gamma_hat)
        fN = RMSE(val_y,solver.Y)
        s = np.where(a_hat !=0)[0].shape[0]
        p = a_hat.shape[0]
        sample_size = solver.N
        MBIC = sample_size * np.log(fN ** 2) + s * np.log(sample_size) * np.log(np.log(p))
        return MBIC

def pack_parameters(Z_train,X_train,Y_train,lmbda_set,lmbda2_set,p1_list,p2_list,d1_list,d2_list,p3_list,d3_list,R_list,na,nb):
    num_lmbda = len(lmbda_set)
    ## the output parameters
    Parameters = []
    for item in range(len(p1_list)):
        p1,p2 = p1_list[item],p2_list[item]
        d1,d2 = d1_list[item],d2_list[item]
        p3,d3 = p3_list[item],d3_list[item]
        ## rerange X
        RX_train = [Rearrange(xi,p1,d1,p2,d2,p3,d3) for xi in X_train]
        for R in R_list:
            for lmbda1 in lmbda_set:
                for lmbda2 in lmbda2_set:
                    Parameters.append((Z_train,RX_train,Y_train,lmbda1,lmbda2,p1,p2,d1,d2,p3,d3,R,na,nb))
    return Parameters


def fun_validate(p1_list,p2_list,d1_list,d2_list,p3_list,d3_list,lmbda_set,lmbda2_set,Z_train,X_train,Y_train,R_list,na,nb,max_iter = 20,print_iter = 5):
    # first we can try Modified BIC as select lambda in SKPD
    train_values = []
    val_values = []
    S = []
    fN_list = []
    solver_list = []
    opt_lmbda_1 = []
    opt_lmbda_2 = []
    for item in range(len(p1_list)):
        p1,p2 = p1_list[item],p2_list[item]
        d1,d2 = d1_list[item],d2_list[item]
        p3,d3 = p3_list[item],d3_list[item]
        ## rerange X

        RX_train = [Rearrange(xi,p1,d1,p2,d2,p3,d3) for xi in X_train]
    #         RX_val = [Rearrange(xi,p1,d1,p2,d2,p3,d3) for xi in X_val]
        # print("\n")
        # print("Iteration p1 x p2 x p3 = %.0f x %.0f x %.0f d1 x d2 x d3 = %.0f x %.0f x %.0f" %(p1,p2,p3,d1,d2,d3))
        # print("\n")
        for R in R_list:
            for lmbda1 in lmbda_set:
                for lmbda2 in lmbda2_set:
                    cv_count = 0
                    # print("\n")
                    # print("Iteration p1 x p2 = %.0f x %.0f d1 x d2 = %.0f x %.0f " %(p1,p2,d1,d2))
                    # print("lambda 1: %f" %lmbda1)
                    # print("lambda 2: %f" %lmbda2)
                    # print("\n")
                        ##### start training
                    solver = AltMin(Z_train,RX_train,Y_train,lmbda1,lmbda2,p1,p2,d1,d2,p3,d3,R,na,nb)
                    a_hat,b_hat,gamma_hat,Y_hat,err_beta,fN = solver.fit(max_iter = max_iter,iter_print = print_iter)

                    if fN != np.Inf:  
                        _,_,kron_ab = func_kron_ab(a_hat,b_hat,R,p1,p2,d1,d2,p3,d3)
                        val_y = solver.fun_predict(Z_train,solver.RX,a_hat,b_hat,gamma_hat)
                        fN = RMSE(val_y,solver.Y)
                        # the number of non-zero entry
                        s = np.where(a_hat !=0)[0].shape[0]
                        p = a_hat.shape[0]
                        sample_size = solver.N
                        MBIC = sample_size * np.log(fN ** 2) + s * np.log(sample_size) * np.log(np.log(p))
                        val_values.append(MBIC)
                        opt_lmbda_1.append(lmbda1)
                        opt_lmbda_2.append(lmbda2)
                        S.append(s)
                    else:
                        val_values.append(np.Inf)
    return val_values