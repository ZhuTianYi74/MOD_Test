from galaxy_mnist import GalaxyMNIST, GalaxyMNISTHighrez
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
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
from GMMD import *

def load_images_list(highres):
    if highres:
        dataset = GalaxyMNISTHighrez(  # [3, 224, 224]
            root='./galaxy_data',
            download=True,
            train=False,
        )
    else:
        dataset = GalaxyMNIST(  # [3, 64, 64]
            root='./galaxy_data',
            download=True,
            train=False,
        )

    (custom_train_images, custom_train_labels), (custom_test_images, custom_test_labels) = dataset.load_custom_data(test_size=0.5, stratify=True) 
    images = torch.cat((custom_train_images, custom_test_images))
    labels = torch.cat((custom_train_labels, custom_test_labels))

    images_list = (
        images[labels == 3].numpy(),
        images[labels == 2].numpy(),
        images[labels == 1].numpy(),
        images[labels == 0].numpy(),    
    )
    
    return images_list

def sampler_galaxy(seed, m, n, corruption, images_list):
    """
    For X: we sample uniformly from images with labels 3, 2, 1.
    For Y: with probability 'corruption' we sample uniformly from images with labels 3, 2, 1.
           with probability '1 - corruption' we sample uniformly from images with labels 0.
    """
    np.random.seed(seed)
    images_0, images_1, images_2, images_3 = images_list
    
    # X
    choice = np.random.choice(3, size=m, replace=True)
    m_0 = np.sum(choice == 0)  # m = m_0 + m_1 + m_2
    m_1 = np.sum(choice == 1)
    m_2 = np.sum(choice == 2)
    indices_0 = np.random.permutation(np.arange(images_0.shape[0]))[:m_0]
    indices_1 = np.random.permutation(np.arange(images_1.shape[0]))[:m_1]
    indices_2 = np.random.permutation(np.arange(images_2.shape[0]))[:m_2]
    X = np.concatenate((images_0[indices_0], images_1[indices_1], images_2[indices_2]), axis=0)
    np.random.shuffle(X)
        
    # Y
    choice = np.random.choice(4, size=n, replace=True, p=[(1-corruption) / 3, (1-corruption) / 3, (1-corruption) / 3, corruption])
    n_0 = np.sum(choice == 0)  # n = n_0 + n_1 + n_2 + n_3
    n_1 = np.sum(choice == 1)
    n_2 = np.sum(choice == 2)
    n_3 = np.sum(choice == 3)
    indices_0 = np.random.permutation(np.arange(images_0.shape[0]))[:n_0]
    indices_1 = np.random.permutation(np.arange(images_1.shape[0]))[:n_1]
    indices_2 = np.random.permutation(np.arange(images_2.shape[0]))[:n_2]
    indices_3 = np.random.permutation(np.arange(images_3.shape[0]))[:n_3]
    Y = np.concatenate((images_0[indices_0], images_1[indices_1], images_2[indices_2], images_3[indices_3]), axis=0)
    np.random.shuffle(Y)
    
    return X, Y

# Ensure images_list is defined before calling any functions that use it
images_list = load_images_list(highres=False)

def fun(sets, n, kernel_type):
    np.random.seed(sets)
    
    # Generate data
    X, Y = sampler_galaxy(sets, m=n, n=n, corruption=0.15, images_list=images_list)
    X = X.reshape((X.shape[0], -1)).astype(np.float32)
    Y = Y.reshape((Y.shape[0], -1)).astype(np.float32)
    
    # Set up kernel functions
    if kernel_type == 'RBF' or kernel_type is None:
        kernel_type = 'RBF'
        bw = get_median_bw(X=X, Y=Y, metric_type="euclidean")
        kernel_func = partial(RBFKernel, bw=bw)
    elif kernel_type == 'Laplace':
        bw = get_median_bw(X=X, Y=Y, metric_type="cityblock")
        kernel_func = partial(LaplaceKernel, bw=bw)
    elif kernel_type == 'Laplace2':
        bw = get_median_bw(X=X, Y=Y, metric_type="euclidean")
        kernel_func = partial(LaplaceKernel2, bw=bw)
    
    # Define the unbiased_mmd2 function to be used with get_bootstrap_threshold
    unbiased_mmd2 = partial(TwoSampleMMDSquared, unbiased=True)

    # Compute various statistical tests
    results = {
        "MODboot": 1.0 * (MOD_stat(X, Y, kernel_func) > MOD_thresh(X, Y, kernel_func, 'bootstrap')),
        "MODperm": 1.0 * (MOD_stat(X, Y, kernel_func) > MOD_thresh(X, Y, kernel_func, 'permutation')),
        "MMDperm": 1.0 * (unbiased_mmd2(X, Y, kernel_func) > get_bootstrap_threshold(X, Y, kernel_func, unbiased_mmd2)),
        "MMDspec": 1.0 * (len(X) * TwoSampleMMDSquared(X, Y, kernel_func) > get_spectral_threshold(X, Y, kernel_func, numNullSamp=200)),
        "lMMD": 1.0 * (BlockMMDSquared(X, Y, kernel_func, b=2, return_sig=True, biased=False)[0] > BlockMMDSquared(X, Y, kernel_func, b=2, return_sig=True, biased=False)[1] * get_normal_threshold(0.05)),
        "cMMD": 1.0 * (crossMMD2sampleUnpaired(X, Y, kernel_func) > get_normal_threshold(0.05)),
        "HMMD": 1.0 * (compute_T_k(X, Y, kernel_func)["T_k"] > get_normal_threshold(0.05)),
        "GF": 1.0 * (gtests(X, Y, perm=0)["pval"]["S_A_appr"] < 0.05)
    }
    
    return list(results.values())

if __name__ == "__main__":
    output_dir = "."
    ns = [20, 50, 100, 200, 500, 1000]
    kernel_types = ["RBF", "Laplace", "Laplace2"]
    
    merge = pd.DataFrame(columns=["kernel_type", "n", "MODboot", "MODperm", "MMDperm", "MMDspec", "lMMD", "cMMD", "HMMD", "GF"])
    
    for kernel in kernel_types:
        for n in ns:
            with Pool(processes=min(int(cpu_count() * 0.4), 60)) as pool:
                res = pool.starmap(fun, [(sets, n, kernel) for sets in range(1, 100)])
            
            res_df = pd.DataFrame(res)
            res_df.to_csv(f"{output_dir}/{n}_{kernel}.csv", index=False)
            
            # Merge results
            merge.loc[len(merge)] = [kernel, n] + list(np.mean(res, axis=0))
            print(f"Completed processing for n = {n} and kernel = {kernel}")
    
    # Save merged results
    merge.to_csv(f"{output_dir}/merge(size).csv", index=False)
    
    # Plot results
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
        
        ax.set_xlabel('n',fontsize=14)
        ax.set_ylabel('Power',fontsize=14)
        ax.set_xticks(np.arange(len(p_original)))
        ax.set_xticklabels(p_original)
        ax.grid(True)
        
        fig.legend(handles, labels, loc='upper center', ncol=4)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        # 添加标题
        plt.title(f'Kernel Type: {kernel}',fontsize=16)
        plt.savefig(f'{output_dir}/{kernel}_plot.png')