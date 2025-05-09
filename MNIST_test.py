import torch
import torchvision
import pickle
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import pandas as pd 
from functools import partial 
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from DGP import *
from MOD import *
from MMD import *
from HMMD import *
from GF import *

# load MNIST dataset
with open('mnist_7x7.data', 'rb') as handle:
    MINIST = pickle.load(handle)

# Generate data set P and multiple subsets Q_list
P = np.vstack(
    (MINIST['0'], MINIST['1'], MINIST['2'], MINIST['3'], MINIST['4'], MINIST['5'], MINIST['6'], MINIST['7'], MINIST['8'],MINIST['9']))
Q1 = np.vstack((MINIST['1'], MINIST['3'], MINIST['5'], MINIST['7'], MINIST['9']))
Q2 = np.vstack((MINIST['0'], MINIST['1'], MINIST['3'], MINIST['5'], MINIST['7'], MINIST['9']))
Q3 = np.vstack((MINIST['0'], MINIST['1'], MINIST['2'], MINIST['3'], MINIST['5'], MINIST['7'], MINIST['9']))
Q4 = np.vstack((MINIST['0'], MINIST['1'], MINIST['2'], MINIST['3'], MINIST['4'], MINIST['5'], MINIST['7'], MINIST['9']))
Q5 = np.vstack((MINIST['0'], MINIST['1'], MINIST['2'], MINIST['3'], MINIST['4'], MINIST['5'], MINIST['6'], MINIST['7'], MINIST['9']))
Q_list = [Q1, Q2, Q3, Q4, Q5]
# Define the kernel function
def fun(P,Q,sets,n,kernel_type):
    np.random.seed(sets)
    
    #generate data
    rs = np.random.RandomState()
    idx_X = rs.randint(len(P), size=n)
    X = P[idx_X, :]
    idx_Y = rs.randint(len(Q), size=n)
    Y = Q[idx_Y, :]
    
    #set up kernel functions
    if kernel_type=='RBF' or kernel_type is None:
        kernel_type='RBF' # just in case it is None
        bw = get_median_bw(X=X, Y=Y,metric_type="euclidean")
        kernel_func = partial(RBFKernel, bw=bw)
    elif kernel_type=='Laplace':
        bw = get_median_bw(X=X, Y=Y,metric_type="cityblock")
        kernel_func = partial(LaplaceKernel, bw=bw)
    elif kernel_type=='Laplace2':
        bw = get_median_bw(X=X, Y=Y,metric_type="euclidean")
        kernel_func = partial(LaplaceKernel2, bw=bw)    
       
    #MOD
    MODstat=MOD_stat(X,Y,kernel_func)
    MODboot=MOD_thresh(X,Y,kernel_func,'bootstrap')
    MODperm=MOD_thresh(X,Y,kernel_func,'permutation')
    MODboot_res=1.0*(MODstat>MODboot)
    MODperm_res=1.0*(MODstat>MODperm)
    
     
    #set up function handles for different threshold computing methods
    thresh_permutation = get_bootstrap_threshold
    thresh_normal = get_normal_threshold
    thresh_spectral = partial(get_spectral_threshold,  numNullSamp=200)
    #mmd-perm
    unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True) 
    mmd_perm_stat = unbiased_mmd2(X, Y, kernel_func)
    mmd_perm_th = thresh_permutation(X, Y, kernel_func, unbiased_mmd2)
    mmd_perm_res=1.0*(mmd_perm_stat>mmd_perm_th)
    #mmd-spectral
    biased_mmd2 = TwoSampleMMDSquared 
    mmd_spectral_stat = len(X)*biased_mmd2(X, Y, kernel_func)
    mmd_spectral_th = thresh_spectral(X, Y, kernel_func)
    mmd_spectral_res=1.0*(mmd_spectral_stat>mmd_spectral_th)
    #l-mmd
    linear_mmd2 = partial(BlockMMDSquared, b=2, return_sig=True, biased=False)
    linear_mmd2_stat, linear_mmd2_sig = linear_mmd2(X, Y, kernel_func)
    linear_mmd2_th = linear_mmd2_sig*thresh_normal(0.05)
    linear_mmd2_res=1.0*(linear_mmd2_stat>linear_mmd2_th)
    # #b-mmd
    # block_mmd2 = partial(BlockMMDSquared, b=max(2, int(sqrt(len(X)))),
    #                                         return_sig=True, biased=False)
    # b_mmd_stat, b_mmd_sig = block_mmd2(X, Y, kernel_func)
    # b_mmd_th = b_mmd_sig*thresh_normal(0.05)
    # b_mmd_res=1.0*(b_mmd_stat>b_mmd_th)
    #c-mmd
    cross_mmd2 = crossMMD2sampleUnpaired
    c_mmd_stat = cross_mmd2(X, Y, kernel_func)
    c_mmd_th = thresh_normal(0.05)
    c_mmd_res=1.0*(c_mmd_stat>c_mmd_th)
    
    #HMMD
    res_HMMD=compute_T_k(X, Y, kernel_func)
    hmmd_th=thresh_normal(0.05)
    HMMD_res=1.0*(res_HMMD["T_k"]>hmmd_th)
    
    #GF
    res_GF=gtests(X,Y,perm=0)
    GF_res=1.0*(res_GF["pval"]["S_A_appr"]<0.05)

   
    res=[MODboot_res,MODperm_res,
         mmd_perm_res,mmd_spectral_res,
         linear_mmd2_res,c_mmd_res,
         HMMD_res,GF_res]
    
    return res
if __name__ == "__main__":
    output_dir = "."
    ns=[20,50,100,200]
    kernel_types=["RBF","Laplace","Laplace2"]
    
    merge = pd.DataFrame(columns=["kernel_type","n",
                                      "MODboot","MODperm","MMDperm",
                                      "MMDspec","lMMD","cMMD",
                                      "HMMD","GF"])
    
    for l in [0,1,2]:
        for i in [0,1,2,3]:
            with Pool(processes=min(int(cpu_count() * 0.7),int(60))) as pool:
                res = pool.starmap(fun, [(P,Q1,sets, ns[i],kernel_types[l]) for sets in range(1, 50)])
                        
            res = pd.DataFrame(res)
            res.to_csv(f"{output_dir}/{ns[i]}_{kernel_types[l]}.csv", index=False)
                    
                    #merge results
            merge.loc[len(merge)]=[kernel_types[l],ns[i]]+list(np.mean(res, axis=0))
            print(f"Completed processing for ns[{i}] = {ns[i]} and kernel_types[{l}] = {kernel_types[l]}")
    #######save results
    merge.to_csv(f"{output_dir}/merge(size).csv", index=False)

    n_methods = 8 
    cmap = plt.get_cmap('tab10', n_methods)
    for kernel in kernel_types:
        kernel_data = merge[merge['kernel_type'] == kernel]
        fig, ax = plt.subplots(figsize=(12, 10))
        handles = []
        labels = []
        p_original = kernel_data['n'].unique() 
        p_mapped = np.linspace(0, len(p_original) - 1, len(p_original)) 
        
        for j, method in enumerate(merge.columns[2:]):
            line, = ax.plot(p_mapped, kernel_data[method], label=f'{method}', color=cmap(j), marker='o')
            handles.append(line)
            labels.append(f'{method}')
                 
        ax.set_xlabel('n')
        ax.set_ylabel('Power')
        ax.set_xticks(np.arange(len(p_original)))
        ax.set_xticklabels(p_original)
        ax.grid(True)
        
        fig.legend(handles, labels, loc='upper center', ncol=4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9) 

        plt.savefig(f'{output_dir}/{kernel}_Q1_plot.png')