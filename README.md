# MOD: kernel-based maximum-of-difference

This project implements our proposed **kernel-based maximum-of-difference (MOD)** testing method for two-sample testing problems, along with several comparison methods. Specifically, we maximize the squared differences between the average distance of the within sample and the between samples across all observations. Accordingly, the proposed test is a max-of-difference type test and can effectively capture subtle differences between two samples.
---

## Project Structure

###  Core Implementation

- `MOD.py`: Our implementation of the kernel-based maximum-of-difference test with two threshold computation approaches:
  - **MOD-bootstrap**: Test with bootstrapped  
  - **MOD-permutation**: Test with permutated critical value

## Comparison Methods

We compare our MOD methods with several existing approaches:

- `MMD.py`: Various MMD variants:`MMD-permutation`, `MMD-spectral`(Gretton et al., 2009), `linear MMD test (lMMD)`(Gretton et al., 2012), `cross MMD test (c-MMD)`(Shekhar et al., 2022)  
- `HMMD.py`: high-dimensional MMD test (Gao and Shao, 2023)  
- `GF.py`: Graph-based testing (Chen and Friedman, 2017) 


---

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

- `cifar_data`: Contains CIFAR-10 and CIFAR-10.1 datasets (https://tensorflow.google.cn/datasets/catalog/cifar10)
- `galaxy_data`: Contains GalaxyMNIST dataset (https://github.com/mwalmsley/galaxy_mnist)
- `mnist_7x7.data`: Contains MNIST dataset

---

##  Usage
##  Simulation
### Test power vs. number of observations and dimension.
```bash
python simu.py
```

### Test power vs. strength of signals.
```bash
python simu2.py
```

### Test power vs. skewness of observations
```bash
python simu4.py
```

### Test power vs. type of kernel
```bash
python simu3.py
```

## Real data
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

## References
[1] Gretton, A., K. Fukumizu, Z. Harchaoui, and B. K. Sriperumbudur (2009). A fast, consistent kernel two-sample test. Advances in neural information processing systems 22.  
[2] Gretton, A., K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola (2012). A kernel two-sample test. The Journal of Machine Learning Research 13(1), 723–773.  
[3] Shekhar, S., I. Kim, and A. Ramdas (2022). A permutation-free kernel two-sample test. Advances in Neural Information Processing Systems 35, 18168–18180.  
[4] Gao, H. and X. Shao (2023). Two sample testing in high dimension via maximum mean discrepancy. Journal of Machine Learning Research 24(304), 1–33.  
[5] Chen, H. and J. H. Friedman (2017). A new graph-based two-sample test for multivariate and object data. Journal of the American statistical association 112(517), 397–409.
