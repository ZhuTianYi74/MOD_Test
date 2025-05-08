import numpy as np
from scipy.linalg import inv, LinAlgError
from numpy.linalg import matrix_rank
import itertools
from scipy.stats import chi2
from scipy.sparse.csgraph import minimum_spanning_tree as mst 
from scipy.spatial.distance import cdist

def gtestsmulti(E, data_list, perm=0):
    K = len(data_list)
    n = np.array([len(data) for data in data_list])
    N = sum(n)
    ind = np.arange(N)

    R = getR(E, n, ind)

    Ebynode = [[] for _ in range(N)]
    for i in range(len(E)):
        Ebynode[E[i, 0]].append(E[i, 1])
        Ebynode[E[i, 1]].append(E[i, 0])

    nE = len(E)
    nodedeg = [len(Ebynode[i]) for i in range(N)]
    nEi = sum(nodedeg[i] * (nodedeg[i] - 1) for i in range(N))  # pair of nodes sharing a node * 2

    muo = np.zeros(K)
    sdo = np.zeros(K)

    p1 = np.array(n)  * (np.array(n)  - 1) / N / (N - 1)
    p2 = p1 * (np.array(n) - 2) / (N - 2)
    p3 = p2 * (np.array(n)  - 3) / (N - 3)

    quan = nE * (nE - 1) - nEi
    muo = nE * p1
    sdo = np.sqrt(nEi * p2 + quan * p3 + muo - muo**2)

    mu = np.eye(K)
    cov11 = np.eye(K)  # covariance function of R_{ii}
    for i in range(K - 1):
        for j in range(i + 1, K):
            mu[i, j] = nE / (N - 1)* 2 * n[i] / N  * n[j]
            cov11[i, j] = (quan * p1[i]/ (N - 2) * n[j] * (n[j] - 1)  / (N - 3) - nE**2 * p1[i] * p1[j])
    
    lower_indices = np.tril_indices(cov11.shape[0], k=0)
    cov11[lower_indices] = cov11.T[lower_indices]
    np.fill_diagonal(cov11, sdo**2)
    np.fill_diagonal(mu, muo)

    R[np.tril_indices(K, -1)] = R.T[np.tril_indices(K, -1)]  # edge matrix
    mu[np.tril_indices(K, -1)] = mu.T[np.tril_indices(K, -1)]  # mean matrix

    # covariance of R_{ii} and R_{jk}
    num = K * (K - 1) // 2 
    cov12 = np.zeros((K, num))
    temp = np.array(list(itertools.combinations(range(K), 2)))
   
    for i in range(K):
        temp1 = np.full((num, 2), i)
        temp2 = [np.setdiff1d(temp[x], temp1[x]) for x in range(num)]
        for j in range(num):
            temp3 = temp2[j]
            if len(temp3) == 1:
                cov12[i, j] = (nEi * n[i] / N / (N - 1)* (n[i] - 1) * n[temp3[0]]  / (N - 2) + 
                               2 * quan / N / (N - 1) * n[i] * (n[i] - 1)/ (N - 2) * (n[i] - 2) * n[temp3[0]]  / (N - 3) -
                               mu[i, i] * mu[i, temp3[0]])
            else:
                cov12[i, j] = (quan * 2 / N / (N - 1)* n[i] * (n[i] - 1) * np.prod(n[temp3])  / (N - 2) / (N - 3) -
                               mu[i, i] * mu[temp3[0], temp3[1]])

   

    # variance of R_{ij}
    sdo1 = np.zeros(num)
    for i in range(num):
        temp1 = n[temp[i]]
        temp2 = np.prod(temp1)
        temp3 = np.sum(temp1)
        temp4 = mu[np.tril_indices(K, -1)]**2
        sdo1[i] = (nE * 2 / (N - 1)* temp2 / N  + nEi  / N / (N - 1)* temp2 * (temp3 - 2) / (N - 2) +
                    quan/ N / (N - 1)  * 4 * temp2 / (N - 2)* (temp2 - temp3 + 1)  / (N - 3) - temp4[i])

    # covariance of R_{ij} and R_{kl}
    temp1 = np.prod(temp, axis=1) 
    temp2 = np.sum(temp, axis=1)  
    temp3 = temp1 - temp2 + 1
    cov22 = np.zeros((num, num))
    for i in range(num):
        u = temp[i, :]
        for j in range(i, num):
            v = temp[j, :]
            temp1 = np.intersect1d(u, v)
            temp2 = np.union1d(u, v)
            if len(temp2) == 3:
                cov22[i, j] = (nEi/ N / (N - 1) * np.prod(n[temp2])  / (N - 2) + 
                               quan * 4 / N / (N - 1)* np.prod(n[temp2]) * (n[temp1] - 1)  / (N - 2) / (N - 3) -
                               mu[u[0], u[1]] * mu[v[0], v[1]])
            else:
                cov22[i, j] = (quan * 4 / N / (N - 1)* np.prod(n[temp2])  / (N - 2) / (N - 3) -
                               mu[u[0], u[1]] * mu[v[0], v[1]])

    cov22 = cov22 + cov22.T - np.diag(cov22.diagonal())
    np.fill_diagonal(cov22, sdo1)

    Sig = np.block([[cov11, cov12], [np.zeros((num, K)), cov22]])
    Sig[np.tril_indices(Sig.shape[0], -1)] = Sig.T[np.tril_indices(Sig.shape[0], -1)]  # covariance matrix of whole R_{ii} and R_{ij}
    KK = Sig.shape[0]

    # SW and SB
    try:
        inv_cov11 = inv(cov11)
    except LinAlgError:
        inv_cov11 = np.linalg.pinv(cov11)
    
    Sw = (np.diag(R) - np.diag(mu)) @ inv_cov11 @ (np.diag(R) - np.diag(mu))
    S_W = Sw

    try:
        inv_cov22 = inv(cov22)
    except LinAlgError:
        inv_cov22 = np.linalg.pinv(cov22)

    vec1 = R[np.triu_indices(K, 1)] - mu[np.triu_indices(K, 1)]
    Sb = vec1 @ inv_cov22 @ vec1
    S_B = Sb

    S = S_W + S_B

    cov33 = Sig[:KK - 1, :KK - 1]
    try:
        inv_cov33 = inv(cov33)
    except LinAlgError:
        inv_cov33 = np.linalg.pinv(cov33)

    vec2 = np.concatenate([np.diag(R), R[np.triu_indices(K, 1)][:-1]]) - np.concatenate([np.diag(mu), mu[np.triu_indices(K, 1)][:-1]])
    Sa = vec2 @ inv_cov33 @ vec2
    S_A = Sa

    # p-values
    S_W_pval_appr = 1 - chi2.cdf(S_W, df=K)
    S_B_pval_appr = 1 - chi2.cdf(S_B, df=matrix_rank(cov22))
    S_pval_appr = 2 * min(S_W_pval_appr, S_B_pval_appr)
    S_A_pval_appr = 1 - chi2.cdf(S_A, df=matrix_rank(cov33))

    result = {'teststat': {'S': S, 'S_A': S_A}, 'pval': {'S_appr': min(1, S_pval_appr), 'S_A_appr': min(1, S_A_pval_appr)}}

    # Permutation test
    if perm > 0:
        S_W_perm = S_B_perm = S_perm = S_A_perm = np.zeros(perm)
        for i in range(perm):
            sam = np.random.permutation(N)
            R_perm = getR(E, n, sam)
            R_perm[np.tril_indices(K, -1)] = R_perm.T[np.tril_indices(K, -1)]

            Sw_perm = (np.diag(R_perm) - np.diag(mu)) @ inv_cov11 @ (np.diag(R_perm) - np.diag(mu))
            S_W_perm[i] = Sw_perm

            vec1_perm = R_perm[np.triu_indices(K, 1)] - mu[np.triu_indices(K, 1)]
            Sb_perm = vec1_perm @ inv_cov22 @ vec1_perm
            S_B_perm[i] = Sb_perm

            S_perm[i] = S_W_perm[i] + S_B_perm[i]

            vec2_perm = np.concatenate([np.diag(R_perm), R_perm[np.triu_indices(K, 1)][:-1]]) - np.concatenate([np.diag(mu), mu[np.triu_indices(K, 1)][:-1]])
            Sa_perm = vec2_perm @ inv_cov33 @ vec2_perm
            S_A_perm[i] = Sa_perm

        S_pval_perm = np.sum(S_perm >= S) / perm
        S_A_pval_perm = np.sum(S_A_perm >= S_A) / perm

        result['pval']['S_perm'] = min(1, S_pval_perm)
        result['pval']['S_A_perm'] = min(1, S_A_pval_perm)

    return result


def getR(E, n, ind):
    K = len(n)
    R = np.zeros((K, K))
    E_ind = np.repeat(np.arange(K), n)
    for i in range(len(E)):
        e1 = E_ind[ind == E[i, 0]]
        e2 = E_ind[ind == E[i, 1]]
        R[e1, e2] += 1
    
    R = np.triu(R) + np.triu(R.T,k=1)
   
    return R

def gtests(X,Y,perm=0):

    data_list=[X,Y]
    
    Z = np.vstack((X, Y))
    D = cdist(Z, Z)
    M = 1.0*(mst(D)>0)
    row_indices, col_indices = np.nonzero(M)
    E= np.vstack((row_indices, col_indices)).T
    
    result=gtestsmulti(E, data_list,perm)
    
    return result

def multigtests(datalist,perm=0):

    data=np.vstack(datalist)
    D = cdist(data, data)
    M = 1.0*(mst(D)>0)
    row_indices, col_indices = np.nonzero(M)
    E= np.vstack((row_indices, col_indices)).T
    
    result=gtestsmulti(E, datalist,perm)
    
    return result
