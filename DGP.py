import numpy as np
from scipy.stats import multivariate_normal,multivariate_t

def dgpK_unb(n, p, ratio, param, null_type):
    X=multivariate_normal.rvs(mean=np.zeros(p), cov=np.eye(p), size=round(ratio*n))
    
    if null_type == "mean":
        Y = multivariate_normal.rvs(mean=np.full(p, param), cov=np.eye(p), size=round((1-ratio)*n))
    elif null_type == "cov":
        cov_matrix = np.eye(p) + param * np.ones((p, p))
        Y = multivariate_normal.rvs(mean=np.zeros(p), cov=cov_matrix, size=round((1-ratio)*n))
    elif null_type == "loc":
        cov_matrix = np.eye(p) + param * np.ones((p, p))
        Y = multivariate_normal.rvs(mean=np.full(p, param/2.0), cov=cov_matrix, size=round((1-ratio)*n))
    elif null_type == "dstr":
        Y=multivariate_t.rvs(loc=np.zeros(p), shape=np.eye(p), size=round((1-ratio)*n),df=param)
    return X,Y

   



def dgpK2_unb(n, p, ratio, param, null_type):
    
    Sigma = np.zeros((p, p))
    for i in range(p):
        Sigma[i, :] = 0.4 ** np.abs(np.arange(1, p + 1) - (i + 1))
    
    X=multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=round(ratio*n),df=10)
    
    if null_type == "mean":
        Y= multivariate_t.rvs(loc=np.full(p, param), shape=Sigma, size=round((1-ratio)*n),df=10)
    elif null_type == "cov":
        Y=multivariate_t.rvs(loc=np.zeros(p), shape=(1+param)*Sigma, size=round((1-ratio)*n),df=10)
    elif null_type == "loc":
        Y = multivariate_t.rvs(loc=np.full(p, param/2), shape=(1+param)*Sigma, size=round((1-ratio)*n),df=10)
    elif null_type == "dstr":
        Y=multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=round((1-ratio)*n),df=param)
    return X,Y
  
def dgpK3_unb(n, p, ratio, param, null_type):
    
    Sigma = np.zeros((p, p))
    for i in range(p):
        Sigma[i, :] = 0.4 ** np.abs(np.arange(1, p + 1) - (i + 1))
        
    X = np.zeros((round(n * ratio), p))
    index = np.random.binomial(1, 0.7, size=round(n * ratio))
    X[index == 1, :] = multivariate_normal.rvs(mean=np.zeros(p),cov=Sigma,size=np.sum(index == 1))
    X[index == 0, :] = multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=np.sum(index == 0),df=10)
    
    Y = np.zeros((n - round(n * ratio), p))
    index = np.random.binomial(1, 0.7, size=n - round(n * ratio))
        
    if null_type == "mean":
        Y[index == 1, :] = multivariate_normal.rvs(mean=np.full(p, param),cov=Sigma,size=np.sum(index == 1))
        Y[index == 0, :] = multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=np.sum(index == 0),df=10)
    
    elif null_type == "cov":
        Y[index == 1, :] = multivariate_normal.rvs(mean=np.zeros(p),cov=(1+param)*Sigma,size=np.sum(index == 1))
        Y[index == 0, :] = multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=np.sum(index == 0),df=10)
    
    elif null_type == "loc":
        Y[index == 1, :] = multivariate_normal.rvs(mean=np.full(p, param/2),cov=(1+param)*Sigma,size=np.sum(index == 1))
        Y[index == 0, :] = multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=np.sum(index == 0),df=10)
    
    elif null_type == "dstr":
        Y[index == 1, :] = multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=np.sum(index == 1),df=param)
        Y[index == 0, :] = multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=np.sum(index == 0),df=10)
    
        
    return X,Y

def dgpK4_unb(n, p, ratio, param, null_type):
    
    Sigma = np.zeros((p, p))
    for i in range(p):
        Sigma[i, :] = 0.4 ** np.abs(np.arange(1, p + 1) - (i + 1))
    
    X=np.exp(multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=round(ratio*n)))
   
    if null_type == "mean":
        Y=np.exp(multivariate_normal.rvs(mean=np.full(p, param), cov=Sigma, size=round((1-ratio)*n)))
    elif null_type == "cov":
        Y=np.exp(multivariate_normal.rvs(mean=np.zeros(p), cov=(1+param)*Sigma, size=round((1-ratio)*n)))
    elif null_type == "loc":
        Y=np.exp(multivariate_normal.rvs(mean=np.full(p, param/2), cov=(1+param)*Sigma, size=round((1-ratio)*n)))
    elif null_type == "dstr":
        Y=np.exp(multivariate_t.rvs(loc=np.zeros(p), shape=Sigma, size=round((1-ratio)*n),df=param))
    return X,Y

