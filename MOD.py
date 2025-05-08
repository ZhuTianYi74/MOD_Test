import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def al_con(ng1, ng2, delta1, delta2):
    if delta1==delta2:
        temp=0
    else:
        temp=1 / np.sqrt((1 / (ng1 - 1) + 1 / ng2) * abs(delta1 - delta2+ 1e-10))
    return temp


def Pest1(a):
    n = len(a)
    mu = np.sum(a) / (n - 1)
    delta1 = np.sum(a**2) / (n - 1) - mu**2
    delta2 = ((np.sum(a - mu) + mu) ** 2 - np.sum((a - mu) ** 2) + mu**2) / ((n - 2) * (n - 1))
    
    return {'delta1': delta1, 'delta2': delta2}


def Pfun(A):
    n = A.shape[0]
    
    mu = np.sum(A) / (n * (n - 1))
    delta1 = (np.sum((A - mu)**2) - mu**2 * n) / (n * (n - 1))
    
    delta2 = 0
    for i in range(n):
        temp = np.delete(A[i], i) 
        delta2 += ((np.sum(temp - mu)) ** 2 - np.sum((temp - mu) ** 2)) / (n * (n - 1) * (n - 2))
    
    return {'delta1': delta1, 'delta2': delta2}



############# calculate the test statistics
def MOD(A, index):
    N = len(index)  
    # total observations
    Ns = pd.Series(index).value_counts()  
    # the number of observations in each group
    
    T = []
    for i in range(N):
        Ps = Pest1(A[i, :])
        pi11 = Ps['delta1']
        pi12 = Ps['delta2']
        
        gi = index[i]
        pin = np.sum(A[i, :] * (index == gi)) / (Ns[gi] - 1)
        pout = np.sum(A[i, :] * (index != gi)) / (N - Ns[gi])
        temp = (pin - pout) * al_con(Ns[gi], N - Ns[gi], pi11, pi12) *(pin - pout) * al_con(Ns[gi], N - Ns[gi], pi11, pi12)
        T.append(0 if np.isnan(temp) else temp)

    return max(T)

############### permutation 
def permMOD(A, index,perms=1000):
    stats = []
    
    for _ in range(perms):
        temp_index = np.random.permutation(index)
        stats.append(MOD(A, temp_index))
    
    cval = np.quantile(stats, 0.95)
    return cval


##########calculate the covariance 
def CQ_MOD(index, p11, p12):
    N = len(index)  # total observations
    K = len(np.unique(index))  # the number of groups
    Gs = np.arange(1, K + 1)  # groups
    Ns = pd.Series(index).value_counts()  # the number of observations in each group
    
    Q = np.zeros((N, N))
    for k in Gs:
        Qs = Qblock_MOD(k, Gs, Ns, p11, p12)
        nk = np.where(index == k)[0]
        for i in nk:
            for j in range(N):
                Q[i, j] = 1 if i == j else Qs[int(index[j] - 1)]

    return Q

def Qblock_MOD(gi, Gs, Ns, p11, p12):
    n = Ns.sum()
    
    Q = np.zeros(len(Gs))
    i = 0
    for gj in Gs:
        if gi == gj:
            Q[i] = (al_con(Ns[gi], n - Ns[gi], p11, p12) ** 2 *
                    ((1 / (n - Ns[gi]) + 1 / (Ns[gi] - 1)) * p12 -
                     (3 * p12 - p11) / (Ns[gi] - 1) ** 2))
            i += 1
        else:
            Q[i] = (np.sqrt(Ns[gi] - 1) * np.sqrt(Ns[gj] - 1) * 
                     (p11 - (n + 2) * p12) /
                     np.sqrt(n - Ns[gi]) / np.sqrt(n - Ns[gj]) /
                     (n - 1) / (p11 - p12))
            i += 1

    return Q



################ bootstrap to calculate critical value
def BootMOD(Sigma, N=1000, alpha=0.05):
    data = multivariate_normal.rvs(mean=np.zeros(Sigma.shape[0]), cov=Sigma, size=N)
    Js = np.max(data**2, axis=1)
    cval = np.quantile(Js, 1 - alpha)
    
    return cval



def MOD_stat(X,Y,kernel_func):
    n1=X.shape[0]
    n2=Y.shape[0]
    index=np.concatenate((np.ones(n1), np.full(n2, 2)))
    
    #construct the distance matrix
    Z = np.concatenate([X, Y], axis=0)
    A=kernel_func(Z,Z)
    np.fill_diagonal(A, 0)

    #compute the test statistics
    stat=MOD(A,index)
    
    return stat


def MOD_thresh(X,Y,kernel_func,thresh_method='bootstrap'):
    n1=X.shape[0]
    n2=Y.shape[0]
    index=np.concatenate((np.ones(n1), np.full(n2, 2)))
    
    #construct the distance matrix
    Z = np.concatenate([X, Y], axis=0)
    A=kernel_func(Z,Z)
    np.fill_diagonal(A, 0)

    # obtain the threshold
    if thresh_method=='permutation':
        thresh=permMOD(A,index)
    elif thresh_method=='bootstrap':
        Ps = Pfun(A)
        p11 = Ps['delta1']
        p12 = Ps['delta2']

        Sigma_MOD = CQ_MOD(index, p11, p12)
        thresh=BootMOD(Sigma_MOD)
    
    return thresh
        
