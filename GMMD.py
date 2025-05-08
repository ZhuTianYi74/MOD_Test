import numpy as np
from scipy.stats import norm
from scipy.linalg import inv

def kertests(X, Y, kernel_func, r1=1.2, r2=0.8, perm=0):
    m = X.shape[0]
    n = Y.shape[0]

    Z = np.vstack((X, Y))
    N = Z.shape[0]

    #construct the distance matrix
    K=kernel_func(Z,Z)
    np.fill_diagonal(K, 0)

    Kx = np.sum(K[:m, :m]) / m / (m - 1)
    Ky = np.sum(K[m:, m:]) / n / (n - 1)
    Kxy = np.sum(K[:m, m:]) / m / n

    mu_Kx = np.sum(K) / (N * (N - 1))
    mu_Ky = mu_Kx
    mu_Kxy = mu_Kx

    A = np.sum(K**2)
    B = np.sum(np.sum(K, axis=1)**2) - A
    C = np.sum(K)**2 - 2 * A - 4 * B

    p1 = m * (m - 1) / (N * (N - 1))
    p2 = p1 * (m - 2) / (N - 2)
    p3 = p2 * (m - 3) / (N - 3)

    q1 = n * (n - 1) / (N * (N - 1))
    q2 = q1 * (n - 2) / (N - 2)
    q3 = q2 * (n - 3) / (N - 3)

    var_Kx = (2 * A * p1 + 4 * B * p2 + C * p3) / ((m* (m - 1))**2) - mu_Kx**2
    var_Ky = (2 * A * q1 + 4 * B * q2 + C * q3) / ((n * (n - 1))**2) - mu_Ky**2
    cov_Kx_Ky = C / (N * (N - 1) * (N - 2) * (N - 3)) - mu_Kx * mu_Ky

    # test statistic GPK
    COV = np.array([[var_Kx, cov_Kx_Ky], [cov_Kx_Ky, var_Ky]])
    Sinv = inv(COV)
    kmv = np.array([Kx - mu_Kx, Ky - mu_Ky])
    GPK = float(kmv @ Sinv @ kmv)

    # test statistic Z_D
    u_D = m * (m - 1)
    v_D = -n * (n - 1)
    mean_D = mu_Kx * u_D + mu_Ky * v_D
    var_D = (u_D**2) * var_Kx + (v_D**2) * var_Ky + 2 * u_D * v_D * cov_Kx_Ky
    Z_D = (Kx * u_D + Ky * v_D - mean_D) / np.sqrt(var_D)

    # test statistic Z_W1
    u_W1 = r1 * m / N
    v_W1 = n / N
    mean_W1 = mu_Kx * u_W1 + mu_Ky * v_W1
    var_W1 = var_Kx * u_W1**2 + var_Ky * v_W1**2 + 2 * u_W1 * v_W1 * cov_Kx_Ky
    Z_W1 = (Kx * u_W1 + Ky * v_W1 - mean_W1) / np.sqrt(var_W1)

    # test statistic Z_W2
    u_W2 = r2 * m / N
    v_W2 = n / N
    mean_W2 = mu_Kx * u_W2 + mu_Ky * v_W2
    var_W2 = var_Kx * u_W2**2 + var_Ky * v_W2**2 + 2 * u_W2 * v_W2 * cov_Kx_Ky
    Z_W2 = (Kx * u_W2 + Ky * v_W2 - mean_W2) / np.sqrt(var_W2)

    temp_approx = np.sort(np.array([norm.cdf(-Z_W1), norm.cdf(-Z_W2), 2 * norm.cdf(-abs(Z_D))]))
    fGPK_appr = 3 * np.min(temp_approx)

    temp_approx1 = np.sort(np.array([norm.cdf(-Z_W1), norm.cdf(-Z_W2)]))
    fGPKM_appr = 2 * np.min(temp_approx1)

    fGPK_Simes_appr = np.min(np.array([3 * temp_approx[0], 1.5 * temp_approx[1], temp_approx[2]]))
    fGPKM_Simes_appr = np.min(np.array([2 * temp_approx1[0], temp_approx1[1]]))

    result = {
        'teststat': {
            'GPK': GPK,
            'ZW1': Z_W1,
            'ZW2': Z_W2,
            'ZD': Z_D
        },
        'pval': {
            'fGPK_appr': min(1, fGPK_appr),
            'fGPKM_appr': min(1, fGPKM_appr),
            'fGPK_Simes_appr': min(1, fGPK_Simes_appr),
            'fGPKM_Simes_appr': min(1, fGPKM_Simes_appr)
        }
    }

    if perm > 0:
        temp1 = np.zeros(perm)
        temp2 = np.zeros(perm)
        temp3 = np.zeros(perm)
        temp4 = np.zeros(perm)
        
        for i in range(perm):
            id = np.random.choice(N, size=N, replace=False)
            K_i = K[id][:, id]

            Kx_i = np.sum(K_i[:m, :m]) / m / (m - 1)
            Ky_i = np.sum(K_i[m:, m:]) / n / (n - 1)

            kmv_i = np.array([Kx_i - mu_Kx, Ky_i - mu_Ky])
            GPK_i = float(kmv_i @ Sinv @ kmv_i)

            Z_D_i = (Kx_i * u_D + Ky_i * v_D - mean_D) / np.sqrt(var_D)
            Z_W1_i = (Kx_i * u_W1 + Ky_i * v_W1 - mean_W1) / np.sqrt(var_W1)
            Z_W2_i = (Kx_i * u_W2 + Ky_i * v_W2 - mean_W2) / np.sqrt(var_W2)

            temp1[i] = GPK_i
            temp2[i] = Z_D_i
            temp3[i] = Z_W1_i
            temp4[i] = Z_W2_i
        
        GPK_perm = np.mean(temp1 >= GPK)

        perm_pval_Z_D = 2 * np.mean(temp2 >= abs(Z_D))
        perm_pval_Z_W1 = np.mean(temp3 >= Z_W1)
        perm_pval_Z_W2 = np.mean(temp4 >= Z_W2)

        temp_perm = np.sort(np.array([perm_pval_Z_W1, perm_pval_Z_W2, perm_pval_Z_D]))
        fGPK_perm = 3 * np.min(temp_perm)

        temp_perm1 = np.sort(np.array([perm_pval_Z_W1, perm_pval_Z_W2]))
        fGPKM_perm = 2 * np.min(temp_perm1)

        fGPK_Simes_perm = np.min(np.array([3 * temp_perm[0], 1.5 * temp_perm[1], temp_perm[2]]))
        fGPKM_Simes_perm = np.min(np.array([2 * temp_perm1[0], temp_perm1[1]]))

        result['pval']['GPK_perm'] = min(1, GPK_perm)
        result['pval']['fGPK_perm'] = min(1, fGPK_perm)
        result['pval']['fGPKM_perm'] = min(1, fGPKM_perm)
        result['pval']['fGPK_Simes_perm'] = min(1, fGPK_Simes_perm)
        result['pval']['fGPKM_Simes_perm'] = min(1, fGPKM_Simes_perm)

    return result

