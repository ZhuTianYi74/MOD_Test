import numpy as np

def compute_T_k(X, Y, kernel_func):
    n = X.shape[0]
    m = Y.shape[0]
    Z = np.vstack((X, Y))

    a_st_k=-1.0*kernel_func(Z,Z)

    v_k_Z = compute_v_k(a_st_k) - (-1)**2 / (n + m - 1) / (n + m - 3)
    c_nm = 2 / (n * (n - 1)) + 4 / (n * m) + 2 / (m * (m - 1))
    e_k = compute_e_k(a_st_k, n, m)

    return {
        'v_k': v_k_Z,
        'e_k': e_k,
        'T_k': e_k / np.sqrt(c_nm * v_k_Z)
    }


def u_centered_dist(a_st_k):
    n = a_st_k.shape[0]
    
    A_1 = a_st_k
    A_2 = np.kron(np.ones((n, 1)), np.sum(a_st_k, axis=0).reshape(1, -1)) / (n - 2)
    A_3 = np.kron(np.sum(a_st_k, axis=1).reshape(-1, 1), np.ones((1, n))) / (n - 2)
    A_4 = np.full_like(a_st_k, np.sum(a_st_k) / ((n - 1) * (n - 2)))
    
    return A_1 - A_2 - A_3 + A_4


def compute_v_k(a_st_k):
    A_st_k = u_centered_dist(a_st_k)
    n = A_st_k.shape[0]
    return (np.sum(A_st_k**2) - np.sum(np.diag(A_st_k**2))) / (n * (n - 3))



def compute_e_k(a_st_k, n, m):
    c = np.ones((n + m, n + m)) / (n * m)
    
    c[:n, :n] = -1 / (n * (n - 1))
    c[n:n + m, n:n + m] = -1 / (m * (m - 1))
    
    np.fill_diagonal(c, 0)
    
    return np.sum(c * a_st_k)

