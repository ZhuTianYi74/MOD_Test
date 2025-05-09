import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from functools import partial 
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from DGP import *
from MOD import *
from MMD import *
from HMMD import *
from GF import *



img_size = 32
transform = transforms.Compose(
    [transforms.Resize(img_size),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR10 as P(X)
dataset_test = torchvision.datasets.CIFAR10(root='./cifar_data/cifar10', download=True, train=False, transform=transform)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1000, shuffle=True, num_workers=1)

for i, (imgs, Labels) in enumerate(dataloader_test):
    data_all = imgs
    label_all = Labels
P = data_all.numpy().reshape(len(data_all), -1)

# Load CIFAR10.1 as Q (Y)
data_new = np.load('./cifar_data/cifar10.1_v4_data.npy')
data_T = np.transpose(data_new, [0, 3, 1, 2])
ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
data_T = data_T[ind_M]
TT = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans = transforms.ToPILImage()
data_trans = torch.zeros([len(data_T), 3, img_size, img_size])
data_T_tensor = torch.from_numpy(data_T)
for i in range(len(data_T)):
    d0 = trans(data_T_tensor[i])
    data_trans[i] = TT(d0)
Q = data_trans.numpy().reshape(len(data_trans), -1)

# Define the kernel function
def fun(P, Q, sets, n, kernel_type):
    np.random.seed(sets)
    
    # generate data
    rs = np.random.RandomState()
    idx_X = rs.randint(len(P), size=n)
    X = P[idx_X, :]
    idx_Y = rs.randint(len(Q), size=n)
    Y = Q[idx_Y, :]
    
    # set up kernel functions
    if kernel_type == 'RBF' or kernel_type is None:
        kernel_type = 'RBF'  # just in case it is None
        bw = get_median_bw(X=X, Y=Y, metric_type="euclidean")
        kernel_func = partial(RBFKernel, bw=bw)
    elif kernel_type == 'Laplace':
        bw = get_median_bw(X=X, Y=Y, metric_type="cityblock")
        kernel_func = partial(LaplaceKernel, bw=bw)
    elif kernel_type == 'Laplace2':
        bw = get_median_bw(X=X, Y=Y, metric_type="euclidean")
        kernel_func = partial(LaplaceKernel2, bw=bw)    
       
    # MOD
    MODstat = MOD_stat(X, Y, kernel_func)
    MODboot = MOD_thresh(X, Y, kernel_func, 'bootstrap')
    MODperm = MOD_thresh(X, Y, kernel_func, 'permutation')
    MODboot_res = 1.0 * (MODstat > MODboot)
    MODperm_res = 1.0 * (MODstat > MODperm)
    
    # set up function handles for different threshold computing methods
    thresh_permutation = get_bootstrap_threshold
    thresh_normal = get_normal_threshold
    thresh_spectral = partial(get_spectral_threshold, numNullSamp=200)
    
    # mmd-perm
    unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True) 
    mmd_perm_stat = unbiased_mmd2(X, Y, kernel_func)
    mmd_perm_th = thresh_permutation(X, Y, kernel_func, unbiased_mmd2)
    mmd_perm_res = 1.0 * (mmd_perm_stat > mmd_perm_th)
    
    # mmd-spectral
    biased_mmd2 = TwoSampleMMDSquared 
    mmd_spectral_stat = len(X) * biased_mmd2(X, Y, kernel_func)
    mmd_spectral_th = thresh_spectral(X, Y, kernel_func)
    mmd_spectral_res = 1.0 * (mmd_spectral_stat > mmd_spectral_th)
    
    # l-mmd
    linear_mmd2 = partial(BlockMMDSquared, b=2, return_sig=True, biased=False)
    linear_mmd2_stat, linear_mmd2_sig = linear_mmd2(X, Y, kernel_func)
    linear_mmd2_th = linear_mmd2_sig * thresh_normal(0.05)
    linear_mmd2_res = 1.0 * (linear_mmd2_stat > linear_mmd2_th)
    
    # c-mmd
    cross_mmd2 = crossMMD2sampleUnpaired
    c_mmd_stat = cross_mmd2(X, Y, kernel_func)
    c_mmd_th = thresh_normal(0.05)
    c_mmd_res = 1.0 * (c_mmd_stat > c_mmd_th)
    
    # HMMD
    res_HMMD = compute_T_k(X, Y, kernel_func)
    hmmd_th = thresh_normal(0.05)
    HMMD_res = 1.0 * (res_HMMD["T_k"] > hmmd_th)
    
    # GF
    res_GF = gtests(X, Y, perm=0)
    GF_res = 1.0 * (res_GF["pval"]["S_A_appr"] < 0.05)
    
    res = [MODboot_res, MODperm_res,
           mmd_perm_res, mmd_spectral_res,
           linear_mmd2_res, c_mmd_res,
           HMMD_res, GF_res]
    
    return res

if __name__ == "__main__":
    output_dir = "."
    ns = [20,50, 100, 200, 500, 1000]  
    kernel_types = ["RBF", "Laplace", "Laplace2"]  
    
    merge = pd.DataFrame(columns=["kernel_type", "n",
                                  "MODboot", "MODperm", "MMDperm",
                                  "MMDspec", "lMMD", "cMMD",
                                  "HMMD", "GF"])
    
    for l in [0, 1, 2]:
        for i in range(len(ns)):
            with Pool(processes=min(int(cpu_count() * 0.7), int(60))) as pool:
                res = pool.starmap(fun, [(P, Q, sets, ns[i], kernel_types[l]) for sets in range(201, 400)])
            
            res = pd.DataFrame(res)
            res.to_csv(f"{output_dir}/{ns[i]}_{kernel_types[l]}.csv", index=False)
                        
            # merge results
            merge.loc[len(merge)] = [kernel_types[l], ns[i]] + list(np.mean(res, axis=0))
            print(f"Completed processing for ns[{i}] = {ns[i]} and kernel_types[{l}] = {kernel_types[l]}")
    
    # save results
    merge.to_csv(f"{output_dir}/Q1_merge.csv", index=False)

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
            line, = ax.plot(p_mapped, kernel_data[method].to_numpy(), label=f'{method}', color=cmap(j), marker='o')
            handles.append(line)
            labels.append(f'{method}')
                 
        ax.set_xlabel('n',fontsize=14)
        ax.set_ylabel('Power',fontsize=14)
        ax.set_xticks(np.arange(len(p_original)))
        ax.set_xticklabels(p_original)
        ax.grid(True)
        
        fig.legend(handles, labels, loc='upper center', ncol=4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9) 

        plt.title(f'Kernel Type: {kernel}',fontsize=16)

        plt.savefig(f'{output_dir}/{kernel}_plot.png')