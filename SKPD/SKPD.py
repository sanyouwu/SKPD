import numpy as np
import numpy.linalg as la
from scipy.stats import ortho_group
import time
import copy
# Lasso and ridge
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import KFold
from .my_operator import *

class AltMin(object):
    def __init__(self,Z,RX,Y,lmbda_1,lmbda_2,p1,p2,d1,d2,p3,d3,R = 1,na = True, nb = True):
        self.Z = Z  ## covariate matrix
        self.RX = RX # list
        self.Y = Y # list
        self.truth = Y ## never change in the update
        # the number of samples
        self.N = len(Y)
        # the shape of Rearranged matrix X
        self.m,self.n = np.shape(self.RX[0])
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2
        # self.beta_true = beta_true
        self.p1 = p1
        self.d1 = d1
        self.p2 = p2
        self.d2 = d2
        self.p3 = p3
        self.d3 = d3
        # R term
        self.R = R
        self.na = na
        self.nb = nb
    
    @staticmethod
    def shrinkage_operator(X,threshold):
        return np.sign(X) * np.maximum((np.abs(X) - threshold), np.zeros(X.shape))
    @staticmethod
    def RMSE(pred,truth):
        sample_size = len(truth)
        fn = la.norm(np.asarray(np.asarray(pred).squeeze() - np.asarray(truth).squeeze()), 2) / np.sqrt(sample_size)  # RMSPE from Hongtu ZHU
        return fn
    def fun_predict(self,Z,RX,a,b,gamma):
        a = a.reshape(-1,1)
        b = b.reshape(-1,1)
        if self.R == 1:
            Y_hat = [np.dot(np.dot(a.T,xi).squeeze(),b.squeeze()) for xi in RX]
        else:
            Ra = a.reshape(self.R,-1)
            Rb = b.reshape(self.R,-1).T
            A,B,kron_ab = func_kron_ab(a,b,self.R,self.p1,self.p2,self.d1,self.d2,self.p3,self.d3)
            beta_hat = kron_ab[-1]
            Y_hat = [np.trace(np.dot(Ra,Xi).dot(Rb)) for Xi in RX]
        if not Z is None:
            gamma = gamma.reshape(-1,1).squeeze()
            co_Y = list(np.squeeze(Z.dot(gamma)))
            Y_hat = [Y_hat[i] + co_Y[i] for i in range(len(co_Y))]
        return Y_hat

    def normalization(self,vec):
        # normalization funciton for lasso estimate
        t = vec.reshape(self.R,-1)
        norm_t = la.norm(t,2,1).reshape(-1,1)
        for i in range(norm_t.shape[0]):
            if norm_t[i] <= 0.05:
                norm_t[i] = 0.05
        stand_t = t/norm_t
        return stand_t.reshape(-1,1)
        
    def _ridge(self,a):
        a = np.asarray(a).reshape(-1,1)
        if self.R == 1:
            design_X = np.zeros([self.N,self.n])
            for i in range(self.N):
                design_X[i,:] = np.dot(a.T,self.RX[i])
        else:
            design_X = np.zeros([self.N,self.n* self.R])
            for i in range(self.N):
                Ra = a.reshape(self.R,self.m)
                design_X[i,:] = np.dot(Ra,self.RX[i]).reshape(1,-1)
        try:
            ## if norm of ak is not too small
            ridge = Ridge(alpha = self.lmbda_2).fit(design_X,np.asarray(self.Y))
        except:
            ## norm of ak is too small
            flag = 1
            return flag,flag
        else:
            flag = 0
            return flag,ridge.coef_.reshape(-1,1)

    def _lasso(self,b):
        b = np.asarray(b).reshape(-1,1)
        if self.R == 1 :
            design_X = np.zeros([self.N,self.m])
            # construct design matrix X in general regression problem
            for i in range(self.N):
                design_X[i,:] = np.dot(b.T,self.RX[i].T)
        else :
            design_X = np.zeros([self.N,self.m * self.R])
            Rb = b.reshape(self.R,self.n)
            for i in range(self.N):
                design_X[i,:] = np.dot(Rb,self.RX[i].T).reshape(1,-1)
        lasso = Lasso(alpha = self.lmbda_1).fit(design_X,np.asarray(self.Y))
        return lasso.coef_.reshape(-1,1)
        
    def _ols(self):
        regr = LR(fit_intercept =False)
        regr.fit(self.Z,np.array(self.truth).squeeze())

        self.gamma = regr.coef_.reshape(-1,1)
        pred_design = regr.predict(self.Z)
        self.Y = [self.truth[i] - pred_design[i] for i in range(len(self.truth))]
        # return self.z,self.Y
            
    def fit(self,tol = None,max_iter = 100,iter_print = 3):
        tic = time.time()
### OLS for gamma hat
        if not self.Z is None:
            self._ols()
        else:
            self.gamma = 0
        _iter = 0
        _tol = 1e-3 ## always 1e-3
        #### taking block-wise lasso as initialization
        # design_X = np.mean(self.RX, axis = 2)
        # reg = LassoCV(cv=5, random_state=0).fit(design_X, self.Y)
        # ak = reg.coef_.reshape(-1,1)
        # if self.R == 1:
        #     ak  
        # if self.na == True:
        #     ak = ak/la.norm(ak,2)
        
        ### initialization---------------------
        if self.R != 0:
            # print("spectral initialization")
            surro_Y = np.mean([self.Y[i] * self.RX[i] for i in range(self.N)],axis = 0)
            a0,sigma,b0t = la.svd(surro_Y)
            # print("Init finished")
            del surro_Y
        if self.R == 1:
            ak = a0[:,:1].reshape(-1,1)
            del a0
        else:
            # SVD initialization
            ak = a0[:,:self.R].reshape(-1,1)
            del a0

            if self.na == True:
                ak = ak/la.norm(ak,2)
        bk = np.ones(self.n * self.R).reshape(-1,1)
        fN_list = []
        err_beta_list = []
        while _iter < max_iter :
            norm_a_old = la.norm(ak,1)
            norm_b_old = la.norm(bk,2)
            flag,bk = self._ridge(ak)
            ## verfify is lambda_1 too large -> norm ak is near to 0 
            if flag == 1:
                print("stop here, lambda is too large!!!")
                norm_ak = 0
                Y_hat,err_beta_list,fN_list = 0,0,np.Inf
                break
            norm_bk = la.norm(bk,2)
            if norm_bk >= 1000000:
                print("stop here, lambda is too large and norm of bk is too large!!!")
                norm_ak = 0
                self.ak = 0
                self.bk = 0
                Y_hat,err_beta_list,fN_list = 0,0,np.Inf
                break
                
            # print("norm of bk: ",norm_bk)
            ## normalization of bk
            # return_bk = copy.deepcopy(bk)
            if self.nb== True:
                bk = bk/norm_bk
            ak = self._lasso(bk)
            norm_ak = la.norm(ak,2)
            ## normalizaiton of ak
            if self.na == True:
                ak = ak/norm_ak
                return_ak = copy.deepcopy(ak*norm_ak)
            else:
                return_ak = copy.deepcopy(ak)
            # print("self.d1,",self.d1)
            # print("self.d2:",self.d2)
            Y_hat = self.fun_predict(None,self.RX,return_ak,bk,0)
            fN = self.RMSE(Y_hat,self.Y)
            _iter += 1
            ## save the best beta_hat
            if _iter == 1:
                best_fN = fN
                self.ak = return_ak
                self.bk = bk
            else:
                if fN < best_fN:
                    best_fN = copy.deepcopy(fN)
                    self.ak = copy.deepcopy(return_ak)
                    self.bk = copy.deepcopy(bk)
            norm_a_new = la.norm(return_ak,1)  # L1 penalty in calculation of total error
            norm_b_new = la.norm(bk,2)
            if _iter >= 2:
                update_err_t =  copy.deepcopy(err_t) 
            err_t = fN + self.lmbda_1 * norm_a_new + self.lmbda_2 * norm_b_new
            if _iter >= 2:
                diff_err_t = np.abs(err_t - update_err_t)

            err_beta_list.append(err_t)
            fN_list.append(fN)
            
            if (_iter % iter_print) == 0  or _iter > max_iter:
                # print("iteration: %d" %_iter, end = " ")
                # print("t_err: %.10f" %err_t, end = " ")
                # print("f(x): %.10f" %(fN), end = " ")
                # print("|a|_1: %.10f" % (la.norm(return_ak,1)), end = " ")
                # print("|a|_2: %.10f" % (la.norm(return_ak,2)), end = " ")
                # print("|b|_2 : %.10f" % (norm_bk))
                if (norm_a_new == 0) or (norm_b_new == 0) : 
                    print(" tips: penalty too large, please decrease the lambda_1")
                    break
                # if _iter >= 5 and  diff_err_t <= _tol:
                #     print("--------------enough iteration---------------")
                #     break
                # print("\n")
        toc = time.time() - tic
        # print("time used: %f" % toc)
        return self.ak,self.bk,self.gamma,Y_hat,err_beta_list,fN_list        