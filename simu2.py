import numpy as np
import pandas as pd
from functools import partial 
from multiprocessing import Pool, cpu_count
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from DGP import *
from MOD import *
from MMD import *
from HMMD import *
from GF import *


def fun(sets,n,p,ratio,param,null_type,kernel_type):
    np.random.seed(sets)
    
    #generate data
    X, Y = dgpK2_unb(n, p, ratio, param, null_type)
    
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
    ns=[200]
    ps=[200]
    types=["mean","cov","loc","dstr"]
    kernel_types=["RBF","Laplace","Laplace2"]
    
    merge = pd.DataFrame(columns=["kernel_type","type","theta",
                                     "MODboot","MODperm","MMDperm",
                                     "MMDspec","lMMD","cMMD",
                                     "HMMD","GF"])
    
    for i in [0]:
        for l in [0,1,2]:
            for k in [0,1,2,3]:  
                if k==0:
                    thetas=[0.06,0.09,0.12,0.15,0.18,0.21]
                elif k==1:
                    thetas=[0.06,0.09,0.12,0.15,0.18,0.21]
                elif k==2:
                    thetas=[0.06,0.09,0.12,0.15,0.18,0.21]
                elif k==3:
                    thetas=[8,7,6,5,4,3]
                    
                for j in [0,1,2,3,4,5]:
                    with Pool(processes=min(int(cpu_count() * 0.7),int(60))) as pool:
                        res = pool.starmap(fun, [(sets, ns[i], ps[0], 1/2, thetas[j], 
                                                  types[k],kernel_types[l]) for sets in range(1, 201)])
                        
                    res = pd.DataFrame(res)
                    res.to_csv(f"{output_dir}/{ns[i]}_{ps[0]}_{thetas[j]}_{types[k]}_{kernel_types[l]}.csv", index=False)
                    
                   #merge results
                    merge.loc[len(merge)]=[kernel_types[l],types[k],thetas[j]]+list(np.mean(res, axis=0))
    
    
    #######save results
    merge.to_csv(f"{output_dir}/merge.csv", index=False)

    n_methods =8
    cmap = cm.get_cmap('tab10', n_methods)
    
    for kernel in kernel_types:
        kernel_data = merge[merge['kernel_type'] == kernel]
    
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        handles = []
        labels = []

        for i, t in enumerate(types):
            ax = axs[i//2, i%2]  
            type_data = kernel_data[kernel_data['type'] == t]
            p_original = type_data['theta']
            p_mapped = np.linspace(0, len(p_original) - 1, len(p_original)) 
        
            for j, method in enumerate(merge.keys()[3:]):
                line,=ax.plot(p_mapped, type_data[method], label=f'{method}', color=cmap(j), marker='o')

                if i == 0:
                    handles.append(line)
                    labels.append(f'{method}')
                
            ax.set_title(f'{t}')
            ax.set_xlabel('signal')
            ax.set_ylabel('Power')
            ax.set_xticks(np.arange(len(p_original)))
            ax.set_xticklabels(p_original)
            ax.grid(True)
        
        fig.legend(handles, labels,loc='upper center', ncol=5)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9) 

        plt.savefig(f'{output_dir}/{kernel}_plot.png')

                    
                
                    
    