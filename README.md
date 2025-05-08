# MOD: Maximum-of-difference Test

This project implements our proposed **Maximum-of-difference (MOD)** testing method for two-sample testing problems, along with several comparison methods. We focus on kernel-based testing techniques for determining whether two samples come from the same distribution.

---

## Project Structure

###  Core Implementation

- `MOD.py`: Our implementation of the kernel-based maximum-of-difference test with two threshold computation approaches:
  - **MOD-bootstrap**: Using bootstrap for threshold calculation  
  - **MOD-permutation**: Using permutation for threshold calculation

### Comparison Methods

- `MMD.py`: Maximum Mean Discrepancy implementation  
- `HMMD.py`: Hierarchical MMD test implementation  
- `GF.py`: Graph-based testing method  
- `GMMD.py`: Generalized MMD implementation  

### Utility Modules

- `DGP.py`: Data generation process functions  

### Data Processing

- `galaxy_mnist.py`: Module for processing GalaxyMNIST dataset  

###  Test Scripts

- `MNIST_test.py`: Evaluating our MOD methods on MNIST data  
- `CIFAR_10_test.py`: Evaluating our MOD methods on CIFAR-10 data  
- `galaxy_test.py`: Evaluating our MOD methods on GalaxyMNIST data  

###  Simulation Scripts

- `simu.py`, `simu2.py`, `simu3.py`, and `simu4.py`: Testing in different simulation scenarios  

### Data Directories (Due to file size limitations, cifar_data and galaxy_data must be downloaded by yourself)

- `cifar_data`: Contains CIFAR-10 and CIFAR-10.1 datasets  
- `galaxy_data`: Contains GalaxyMNIST dataset
- `mnist_7x7.data`: Contains MNIST dataset

---

##  Our Methods: MOD (Maximum Outer Difference)

Our proposed method includes two variants:

- `MODboot`: Using bootstrap to compute the threshold  
- `MODperm`: Using permutation to compute the threshold  


---

## Comparison Methods

We compare our MOD methods with several existing approaches:

- Various MMD variants: `MMD-permutation`, `MMD-spectral`, `l-MMD`, `c-MMD`  
- `HMMD`: Hierarchical MMD  
- `GF`: Graph-based testing  
- `GMMD`: Generalized MMD  

---
##  Usage

### MNIST Data Test
```bash
python MNIST_test.py
```

### CIFAR-10 Data Test
```bash
python CIFAR_10_test.py
```

### Galaxy Data Test
```bash
python galaxy_test.py
```

---

## Test Parameters

- **Sample sizes (n):** [20, 50, 100, 200, 500, 1000]  
- **Kernel types:** `["RBF", "Laplace", "Laplace2"]`  

---

## Output Results

- Testing power of our MOD methods and comparison methods with different sample sizes and kernel functions  
- CSV files with comparison results  
- Visualized power curves  

---

## Dependencies

The project depends on the following Python libraries:

- `numpy`  
- `pytorch`  
- `scipy`  
- `pandas`  
- `matplotlib`  
- `h5py`  
- `scikit-learn`  
- `torchvision`  
