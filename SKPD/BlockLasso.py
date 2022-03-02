from sklearn.linear_model import Lasso,LassoCV
from  .my_operator import *
from .criterion import *
def BlockLasso(X,Y,m1,m2,n1,n2,d1,d2,lam,data_type = "matrix"):
    # if data_type == "matrix":
    RX = [Rearrange(xi,m1,m2,n1,n2,d1,d2) for xi in X]
    RX = np.asarray(RX)
    design_X = np.mean(RX, axis = 2)
    if isinstance(lam,list):
        sample_size = len(Y)
        MBIC_list = []
        for lmbda in lam:
            reg = Lasso(alpha = lmbda, random_state=0).fit(design_X, Y)
            Y_hat = reg.predict(design_X)
            fN = RMSE(Y_hat,Y)
            a_hat = reg.coef_
            s = np.where(a_hat !=0)[0].shape[0]
            p = a_hat.shape[0]
            MBIC = sample_size * np.log(fN ** 2) + s * np.log(sample_size) * np.log(np.log(p))
            MBIC_list.append(MBIC)

        print("MBIC results: ",MBIC_list)
        opt_idx = MBIC_list.index(min(MBIC_list))
        lam = lam[opt_idx]
        print("choosed lambda is : ",lam, " and its MBIC : ", MBIC_list[opt_idx])

    reg = Lasso(alpha = lam, random_state=0).fit(design_X, Y)
    Y_hat = reg.predict(design_X)
    return reg, Y_hat