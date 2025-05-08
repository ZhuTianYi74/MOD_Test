from math import sqrt
import scipy.stats as stats  
import numpy as np 
from scipy.spatial.distance import cdist, pdist 
from tqdm import tqdm 



def RBFKernel(x, y=None, bw=1.0, amp=1.0):

    y = x if y is None else y 

    dists = cdist(x, y)
    squared_dists = dists * dists 
    k = amp * np.exp( -(1/(2*bw*bw)) * squared_dists ) 
   
    return k 

# def LinearKernel(x, y=None):
#     y = x if y is None else y 
#     k = np.einsum('ji, ki -> jk', x, y)
#     return k 

# def PolynomialKernel(x, y=None, c=1.0, scale=1.0, 
#                         degree=2):
#     y = x if y is None else y
#     # get the matrix of dot-products 
#     D = LinearKernel(x=x, y=y)
#     # compute the polynomial kernel 
#     assert scale!=0
#     k = ((D + c)/scale)**degree 
#     return k 

def LaplaceKernel(x,y=None,bw=1.0,amp=1.0):
    y = x if y is None else y 
    abs_dists=cdist(x,y,metric="cityblock")
    k=amp * np.exp( -(1/bw) * abs_dists ) 
    return k

def LaplaceKernel2(x, y=None, bw=1.0, amp=1.0):

    y = x if y is None else y 

    dists = cdist(x, y)
    k = amp * np.exp( -(1/bw) * dists ) 
   
    return k 

def get_median_bw(Z=None, X=None, Y=None,metric_type="euclidean"):
    if Z is None:
        assert (X is not None) and (Y is not None)
        Z = np.concatenate([X, Y], axis=0)
    dists_ = pdist(Z,metric=metric_type)
    sig = np.median(dists_)
    return sig



def crossMMD2sampleUnpaired(X, Y, kernel_func):
    """
        Compute the studentized cross-MMD statistic
    """
    n, d = X.shape 
    m, d_ = Y.shape 
    # sanity check 
    assert (d_==d) and (n>=2) and (m>=2) 

    n1, m1 = n//2, m//2 
    n1_, m1_ = n-n1, m-m1

    X1, X2 = X[:n1], X[n1:]
    Y1, Y2 = Y[:m1], Y[m1:]

    Kxx = kernel_func(X1, X2) 
    Kyy = kernel_func(Y1, Y2)

    Kxy = kernel_func(X1, Y2) 
    Kyx = kernel_func(Y1, X2)

    # compute the numerator 
    Ux = Kxx.mean() - Kxy.mean()
    Uy = Kyx.mean() - Kyy.mean()  
    U = Ux - Uy
    # compute the denominator 
    term1 = (Kxx.mean(axis=1) - Kxy.mean(axis=1) - Ux)**2
    sigX2 = term1.mean() 
    term2 = (Kyx.mean(axis=1) - Kyy.mean(axis=1) - Uy)**2
    sigY2 = term2.mean() 
    sig = sqrt(sigX2/n1 + sigY2/m1)  
    if not sig>0:
        print(f'term1={term1}, term2={term2}, sigX2={sigX2}, sigY2={sigY2}')
        raise Exception(f'The denominator is {sig}')
    # obtain the statistic
    T = U/sig 
    return T


def TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False, 
                        return_float=False):
    Kxx = kernel_func(X, X) 
    Kyy = kernel_func(Y, Y)
    Kxy = kernel_func(X, Y)

    n, m = len(X), len(Y)

    term1 = Kxx.sum()
    term2 = Kyy.sum()
    term3 = 2*Kxy.mean()

    if unbiased:
        # term1 -= torch.trace(Kxx)
        # term2 -= torch.trace(Kyy)
        term1 -= np.trace(Kxx)
        term2 -= np.trace(Kyy)
        MMD_squared = (term1/(n*(n-1)) + term2/(m*(m-1)) - term3)
    else:
        MMD_squared = term1/(n*n) + term2/(m*m) - term3 
    if return_float:
        return MMD_squared
    else:
        return MMD_squared 
    
    
def BlockMMDSquared(X, Y, kernel_func, b=2, perm=None, biased=True, 
                    return_sig=False):
    # sanity checks 
    n, m = len(X), len(Y) 
    if m>n:
        Y=Y[:n] 
    elif m<n:
        n=m 
        X=X[:n]
    r = n%b 
    if r!=0: # drop the last r terms in X and Y
        n = int(n-r)
        X, Y = X[:n], Y[:n] 
    # now compute the statistic 
    num_blocks = n//b 
    if perm is None:
        perm = np.arange(n) 
    X, Y = X[perm], Y[perm]
    KX, KY, KXY = kernel_func(X), kernel_func(Y), kernel_func(X, Y)
    KYX = kernel_func(Y, X)

    # blockMMD = 0 
    blockMMD = np.zeros((num_blocks,)) 
    sigMMD =  np.zeros((num_blocks,))
    tempK = np.zeros((b,b))
    Z = np.concatenate((X, Y))
    for i in range(num_blocks):
        idx0, idx1 = b*i, b*(i+1)
#        idx = perm[idx0:idx1]
#        Xi, Yi = X[idx], Y[idx] 
#        KX, KY, KXY = kernel_func(Xi), kernel_func(Yi), kernel_func(Xi, Yi)
        if biased:
            tempK += (KX[idx0:idx1, idx0:idx1])
            tempK += (KY[idx0:idx1, idx0:idx1])
            tempK -= (KXY[idx0:idx1, idx0:idx1])
            tempK -= (KYX[idx0:idx1, idx0:idx1])
        else:
            Xi = X[idx0:idx1]
            Yi = Y[idx0:idx1]
            # blockMMD += TwoSampleMMDSquared(Xi, Yi, kernel_func, unbiased=True) 
            blockMMD[i] = TwoSampleMMDSquared(Xi, Yi, kernel_func, unbiased=True) 
            idx1X = np.random.permutation(2*n)[:b]
            idx1Y = np.random.permutation(2*n)[:b]
            sigMMD[i] = TwoSampleMMDSquared(
                X=Z[idx1X], Y=Z[idx1Y], kernel_func=kernel_func, unbiased=True
            )
 
    if biased:
        stat = (1/num_blocks)*(tempK.mean())
        if return_sig:
            raise Exception('Not computed the std for biased version')
        else:
            return stat
    else:
        # stat = (1/num_blocks)*blockMMD
        stat = blockMMD.mean()
        if return_sig:
            sig = (1/sqrt(num_blocks))*sigMMD.std()
            return stat, sig
        else:
            return stat
 
    



def get_bootstrap_threshold(X, Y, kernel_func, statfunc, alpha=0.05,
                            num_perms=1000, progress_bar=False,
                            return_stats=False, use_numpy=False):
    """
        Return the level-alpha rejection threshold for the statistic 
        computed by the function handle stat_func using num_perms 
        permutations. 
    """
    assert len(X.shape)==2
    # concatenate the two samples 
    if use_numpy:
        Z = np.vstack((X,Y))
    else:
        Z = np.vstack((X, Y))
    # assert len(X)==len(Y)
    n,  n_plus_m = len(X), len(Z)
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) # 
    
    original_statistic = statfunc(X, Y, kernel_func)
    if use_numpy:
        perm_statistics = np.zeros((num_perms,))
    else:
        perm_statistics = np.zeros((num_perms,))

    range_ = tqdm(range(num_perms)) if progress_bar else range(num_perms)
    for i in range_:
        if use_numpy:
            perm = np.random.permutation(n_plus_m)
        else:
            perm = np.random.permutation(n_plus_m)
        X_, Y_ = Z[perm[:n]], Z[perm[n:]] 
        stat = statfunc(X_, Y_, kernel_func)
        perm_statistics[i] = stat

    # obtain the threshold
    if use_numpy:
        perm_statistics = np.sort(perm_statistics) 
    else:
        perm_statistics  = np.sort(perm_statistics) 
    i_ = int(num_perms*(1-alpha)) 
    threshold = perm_statistics[i_]
    if not use_numpy:
        threshold = threshold
    if return_stats:
        return threshold, perm_statistics
    else:
        return threshold


def get_normal_threshold(alpha):
    return stats.norm.ppf(1-alpha)

def get_spectral_threshold(X, Y, kernel_func, alpha=0.05, numEigs=None,
                            numNullSamp=200):
    n = len(X)
    assert len(Y)==n

    if numEigs is None:
        numEigs = 2*n-2
    numEigs = min(2*n-2, numEigs)

    testStat = n*TwoSampleMMDSquared(X, Y, kernel_func, unbiased=False)

    #Draw samples from null distribution
    Z = np.vstack((X, Y))
    # kernel matrix of the concatenated data
    KZ = kernel_func(Z, Z) # 
    
    H = np.eye(2*n) - 1/(2*n)*np.ones((2*n, 2*n))
    KZ_ = np.matmul(H, np.matmul(KZ, H))


    kEigs = np.linalg.eigvals(KZ_)[:numEigs]
    kEigs = 1/(2*n) * abs(kEigs); 
    numEigs = len(kEigs);  

    nullSampMMD = np.zeros((numNullSamp,))

    for i in range(numNullSamp):
        samp = 2* np.sum( kEigs * (np.random.randn(numEigs))**2)
        nullSampMMD[i] = samp

    nullSampMMD  = np.sort(nullSampMMD)
    threshold = nullSampMMD[round((1-alpha)*numNullSamp)]
    return threshold


